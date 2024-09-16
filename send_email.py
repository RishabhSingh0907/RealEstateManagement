import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

MAIL_SERVER = "smtp.gmail.com"
MAIL_USERNAME = "abhyasttechnosolution@gmail.com"
APP_PASSWORD = "qrnz bkyy acaj fjci"
SMTP_PORT = 587
print("mail_username",MAIL_USERNAME)
def send_notification(receiver_email, MAIL_USERNAME, TOKEN):
    
    subject = "Forgot Password Request"
    # Create the email message
    body = f"""Dear User,

        Please enter below token to reset the password:

        Token: {TOKEN}
       
    Best regards,
    Heart Disease Detector
    """

    msg = MIMEMultipart()
    msg['From'] = MAIL_USERNAME
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Connect to the SMTP server

    server = smtplib.SMTP(MAIL_SERVER, int(SMTP_PORT))  # Change the server and port if using a different email service
    server.starttls()
    server.login(MAIL_USERNAME, APP_PASSWORD)  # Use the App Password here

    # Send the email
    text = msg.as_string()
    server.sendmail(MAIL_USERNAME, receiver_email, text)

    # Close the SMTP server connection
    server.quit()

    print('Email sent successfully')



if __name__ == "__main__":
    send_notification("wisdomml2020@gmail.com ")