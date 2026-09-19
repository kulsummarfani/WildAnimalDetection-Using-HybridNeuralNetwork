
import os
from pathlib import Path
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

load_dotenv(Path(__file__).resolve().parent / '.env')


def alert_configuration_error() -> str | None:
    """Return a human-readable configuration error, or None when ready."""
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    from_number = os.environ.get('TWILIO_FROM_NUMBER')
    phone_number = os.environ.get('ALERT_PHONE_NUMBER')
    if not account_sid:
        return 'TWILIO_ACCOUNT_SID is missing.'
    if not auth_token:
        return 'TWILIO_AUTH_TOKEN is missing.'
    if not from_number:
        return 'TWILIO_FROM_NUMBER is missing.'
    if not phone_number:
        return 'ALERT_PHONE_NUMBER is missing.'
    if not phone_number.isdigit() or len(phone_number) != 10:
        return 'ALERT_PHONE_NUMBER must be a 10-digit Indian number.'
    return None


def send_sms_alert(species: str, confidence: float) -> bool:
    """Send one SMS alert using configured Twilio credentials.

    Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER (a Twilio
    number, e.g. +1XXXXXXXXXX) and ALERT_PHONE_NUMBER (10-digit Indian number)
    first. Returns False instead of contacting the API when configuration is
    absent.
    """
    configuration_error = alert_configuration_error()
    if configuration_error:
        print(f'SMS alert skipped: {configuration_error}')
        return False

    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    from_number = os.environ['TWILIO_FROM_NUMBER']
    phone_number = os.environ['ALERT_PHONE_NUMBER']

    message_body = (
        f'ALERT: {species} detected ({confidence * 100:.0f}% confidence). '
        'Check camera feed immediately.'
    )

    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=f'+91{phone_number}',
        )
        print(f'SMS alert sent: SID {message.sid}, status {message.status}')
        return True
    except TwilioRestException as exc:
        print(f'Failed to send SMS alert: {exc}')
        return False