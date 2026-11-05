"""Authenticated, bounded replacement for the raw voice REST command.

Registered through the native HA API: a service-data `context` must never
replace the trusted ServiceCall.context, as it can in a pyscript decorator.
"""
from custom_components.pyscript.function import Function


@pyscript_compile
def install():
    hass = Function.hass
    # Stay inside compiled Python: Pyscript DEBUG logs interpreted call args.
    config = globals()["pyscript.app_config"]
    import asyncio
    import time
    import logging
    import uuid
    from homeassistant.core import SupportsResponse
    from homeassistant.exceptions import HomeAssistantError, Unauthorized
    from homeassistant.helpers.aiohttp_client import async_get_clientsession

    # Keep admission state across pyscript reloads. No persisted credentials.
    admission = hass.data.setdefault("voiceconsult_admission", {"until": 0.0, "busy": False})
    session = async_get_clientsession(hass, verify_ssl=True)

    async def consult(call):
        user_id = call.context.user_id
        user = await hass.auth.async_get_user(user_id) if user_id else None
        if user is None or not user.is_active or not user.is_admin:
            raise Unauthorized()
        if set(call.data) != {"question"}:
            raise HomeAssistantError("Voice consult accepts only question")
        question = call.data["question"]
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise HomeAssistantError("Voice question must contain 1 to 2000 characters")
        now = time.monotonic()
        if admission["busy"] or now < admission["until"]:
            raise HomeAssistantError("Voice consult busy; retry after the cooldown")
        # No awaits between checking and reserving. One POST at a time and at
        # most one accepted attempt per minute across all callers, even on error.
        admission["busy"] = True
        admission["until"] = now + 60
        muted = hass.states.is_state("input_boolean.voice_consult_test_mode", "on")
        payload = {
            "message": ("[VOICE-CONSULT-TEST] " if muted else "[VOICE-CONSULT-ASYNC] ")
            + question.strip() + " (asked by voice)",
            "name": "HA-Voice-Consult", "agentId": config.get("agent_id", "voice"),
            "thinking": "low", "deliver": not muted,
            "channel": config.get("channel", "telegram"), "to": config["chat_id"],
            "timeoutSeconds": 45,
        }
        idem = str(uuid.uuid4())
        headers = {"Authorization": config["authorization"], "Idempotency-Key": idem}
        # Replaying the admitted request with the same key joins its completion;
        # it never creates a second turn. Hold the slot after the fast HA ACK.
        async def observe_completion(run_id):
            try:
                async with asyncio.timeout(90):
                    async with session.post(
                        config["url"], json={**payload, "waitForCompletion": True},
                        allow_redirects=False, headers=headers,
                    ) as response:
                        result = await response.json()
                        terminal = result.get("completion", {}).get("status")
                        if (response.status != 200 or result.get("ok") is not True
                                or result.get("runId") != run_id
                                or terminal not in {"ok", "error", "skipped"}):
                            raise HomeAssistantError("Voice completion receipt invalid")
                admission["busy"] = False
                admission["run_id"] = None
            except Exception:
                # Unknown completion fails closed. Never free a slot just because
                # the HTTP client timed out while the Gateway may still be busy.
                logging.getLogger(__name__).error("Voice completion unconfirmed; admission held")

        dispatched = False
        accepted = False
        try:
            async with asyncio.timeout(15):
                dispatched = True
                async with session.post(
                    config["url"], json=payload, allow_redirects=False, headers=headers,
                ) as response:
                    if response.status != 200:
                        # A proxy 5xx can hide an accepted upstream run.
                        if response.status in {400, 401, 403, 404, 413, 429}:
                            dispatched = False
                        raise HomeAssistantError("Voice hook did not accept request")
                    result = await response.json()
                    if result.get("ok") is not True or not isinstance(result.get("runId"), str):
                        raise HomeAssistantError("Voice hook returned no acceptance receipt")
                    admission["run_id"] = result["runId"]
                    admission["task"] = hass.async_create_task(observe_completion(result["runId"]))
                    accepted = True
                    return {"status": 200, "content": result}
        except HomeAssistantError:
            raise
        except Exception:
            raise HomeAssistantError("Voice hook request failed") from None
        finally:
            if not accepted and not dispatched:
                admission["busy"] = False

    # Never also define a raw YAML rest_command with this name: it would bypass these checks.
    hass.services.async_register("rest_command", "claw_voice_consult", consult,
                                 supports_response=SupportsResponse.OPTIONAL)


@time_trigger("startup")
def start():
    install()


@pyscript_compile
def uninstall():
    Function.hass.services.async_remove("rest_command", "claw_voice_consult")


@time_trigger("shutdown")
def stop():
    uninstall()
