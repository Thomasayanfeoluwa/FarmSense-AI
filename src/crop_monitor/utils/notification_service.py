# import logging
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from twilio.rest import Client
# import anyio

# from src.crop_monitor.config.settings import settings


# logger = logging.getLogger(__name__)

# class NotificationService:
#     def __init__(self):
#         # Twilio setup
#         self.twilio_sid = settings.TWILIO_ACCOUNT_SID
#         self.twilio_token = settings.TWILIO_AUTH_TOKEN
#         self.twilio_phone = settings.TWILIO_PHONE_NUMBER
#         self.client = Client(self.twilio_sid, self.twilio_token)
#         self.whatsapp_from = f"whatsapp:{self.twilio_phone}"

#         # Email setup
#         self.smtp_server = settings.SMTP_SERVER
#         self.smtp_port = settings.SMTP_PORT
#         self.smtp_user = settings.SMTP_USER
#         self.smtp_password = settings.SMTP_PASSWORD
#         self.email_from = settings.SMTP_FROM

#     # ---------------- Twilio Alerts ---------------- #
#     async def send_sms(self, to: str, message: str):
#         try:
#             await anyio.to_thread.run_sync(
#                 lambda: self.client.messages.create(
#                     body=message,
#                     from_=self.twilio_phone,
#                     to=to
#                 )
#             )
#             logger.info(f"SMS sent to {to}: {message}")
#         except Exception as e:
#             logger.error(f"Failed to send SMS to {to}: {e}")

#     async def send_whatsapp(self, to: str, message: str):
#         try:
#             await anyio.to_thread.run_sync(
#                 lambda: self.client.messages.create(
#                     body=message,
#                     from_=self.whatsapp_from,
#                     to=f"whatsapp:{to}"
#                 )
#             )
#             logger.info(f"WhatsApp sent to {to}: {message}")
#         except Exception as e:
#             logger.error(f"Failed to send WhatsApp to {to}: {e}")

#     # ---------------- Email Alerts ---------------- #
#     async def send_email(self, to: str, subject: str, message: str):
#         """
#         Async-safe email sending using SMTP.
#         """
#         try:
#             msg = MIMEMultipart()
#             msg["From"] = self.email_from
#             msg["To"] = to
#             msg["Subject"] = subject
#             msg.attach(MIMEText(message, "plain"))

#             def send():
#                 with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
#                     server.starttls()
#                     server.login(self.smtp_user, self.smtp_password)
#                     server.send_message(msg)

#             await anyio.to_thread.run_sync(send)
#             logger.info(f"Email sent to {to}: {subject}")
#         except Exception as e:
#             logger.error(f"Failed to send email to {to}: {e}")

#     # ---------------- Combined Alert ---------------- #
#     async def send_alert(self, to: str, message: str, subject: str = "AI Crop Monitor Alert"):
#         """
#         Send SMS, WhatsApp, and Email alert.
#         """
#         await self.send_sms(to, message)
#         await self.send_whatsapp(to, message)
#         await self.send_email(to, subject, message)







import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import anyio

from src.crop_monitor.config.settings import settings

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        # ---------------- Twilio Setup ---------------- #
        self.twilio_sid = settings.TWILIO_ACCOUNT_SID
        self.twilio_token = settings.TWILIO_AUTH_TOKEN
        self.twilio_phone = settings.TWILIO_PHONE_NUMBER
        self.whatsapp_from = f"whatsapp:{self.twilio_phone}"

        # Only initialize Twilio client if credentials exist
        if self.twilio_sid and self.twilio_token:
            self.client = Client(self.twilio_sid, self.twilio_token)
            logger.info("Twilio client initialized successfully.")
        else:
            self.client = None
            logger.warning("Twilio credentials missing. SMS/WhatsApp alerts disabled.")

        # ---------------- Email Setup ---------------- #
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.email_from = settings.SMTP_FROM

        if not all([self.smtp_server, self.smtp_port, self.smtp_user, self.smtp_password, self.email_from]):
            logger.warning("Incomplete SMTP configuration. Email alerts disabled.")
            self.email_disabled = True
        else:
            self.email_disabled = False

    # ---------------- Twilio Alerts ---------------- #
    async def send_sms(self, to: str, message: str):
        if not self.client:
            logger.warning("Twilio SMS skipped. Client not configured.")
            return
        try:
            await anyio.to_thread.run_sync(
                lambda: self.client.messages.create(
                    body=message,
                    from_=self.twilio_phone,
                    to=to
                )
            )
            logger.info(f"SMS sent to {to}: {message}")
        except Exception as e:
            logger.error(f"Failed to send SMS to {to}: {e}")

    async def send_whatsapp(self, to: str, message: str):
        if not self.client:
            logger.warning("Twilio WhatsApp skipped. Client not configured.")
            return
        try:
            await anyio.to_thread.run_sync(
                lambda: self.client.messages.create(
                    body=message,
                    from_=self.whatsapp_from,
                    to=f"whatsapp:{to}"
                )
            )
            logger.info(f"WhatsApp sent to {to}: {message}")
        except Exception as e:
            logger.error(f"Failed to send WhatsApp to {to}: {e}")

    # ---------------- Email Alerts ---------------- #
    async def send_email(self, to: str, subject: str, message: str):
        if self.email_disabled:
            logger.warning(f"Email to {to} skipped. SMTP not configured.")
            return
        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_from
            msg["To"] = to
            msg["Subject"] = subject
            msg.attach(MIMEText(message, "plain"))

            def send():
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)

            await anyio.to_thread.run_sync(send)
            logger.info(f"Email sent to {to}: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email to {to}: {e}")

    # ---------------- Combined Alert for Farms ---------------- #
    async def send_alert(self, farm_contacts: dict, message: str, subject: str = "AI Crop Monitor Alert"):
        """
        Sends alerts to all contacts (email, SMS, WhatsApp) for a farm.
        farm_contacts example:
        {
            "email": "farmer@example.com",
            "phone": "+2348012345678",
            "whatsapp": "+2348012345678"
        }
        """
        email = farm_contacts.get("email", "")
        phone = farm_contacts.get("phone", "")
        whatsapp = farm_contacts.get("whatsapp", "")

        # Run tasks concurrently for faster response
        async with anyio.create_task_group() as tg:
            if phone:
                tg.start_soon(self.send_sms, phone, message)
            if whatsapp:
                tg.start_soon(self.send_whatsapp, whatsapp, message)
            if email:
                tg.start_soon(self.send_email, email, subject, message)
