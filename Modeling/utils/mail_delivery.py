import os
import email
import imaplib
import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send(from_mail, password, to_mail, subject, body, attachments = None):
    msg = MIMEMultipart()
    msg['From'] = from_mail
    msg['To'] = to_mail
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    if attachments is not None:
        for attachment_file in attachments:
            attachment = open(attachment_file, 'rb')
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment', filename = attachment_file)
            msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(from_mail, password)
    text = msg.as_string()
    server.sendmail(from_mail, to_mail, text)
    server.quit()

def catch_results(username, password, tag, cleaning = False):
    server = 'imap.gmail.com'
    mail = imaplib.IMAP4_SSL(server)
    mail.login(username, password)
    mail.select('inbox')
    data = mail.search(None, 'ALL')
    mail_ids = data[1]
    id_list = mail_ids[0].split()
    if len(id_list) == 0: return
    latest_email_id = int(id_list[-1])
    for i in range(latest_email_id, 0, -1):
        data = mail.fetch(str(i), '(RFC822)')
        for response_part in data:
            arr = response_part[0]
            if isinstance(arr, tuple):
                try:
                    msg = email.message_from_string(str(arr[1], 'utf-8'))
                except UnicodeDecodeError:
                    msg = email.message_from_string(str(arr[1], 'latin-1'))
                
                email_subject = msg['subject']
                if tag not in email_subject: break
                for part in msg.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                    if part.get('Content-Disposition') is None:
                        continue

                    fileName = part.get_filename()
                    fileName = fileName.split('/')
                    sub_directories = len(fileName) - 1
                    for level in range(sub_directories):
                        try: os.chdir(fileName[level])
                        except:
                            os.mkdir(fileName[level])
                            os.chdir(fileName[level])

                    for level in range(sub_directories): os.chdir('..')
                    fileName = '/'.join(fileName)
                    if bool(fileName):
                        filePath = os.path.join(os.getcwd(), fileName)
                        if os.path.isfile(filePath): os.remove(filePath)
                        fp = open(filePath, 'wb')
                        fp.write(part.get_payload(decode=True))
                        fp.close()
                        
        if tag not in email_subject: break
    
    if cleaning: os.system('clear')