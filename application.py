from flask import Flask, request, redirect, jsonify
from flask_cors import CORS
from flasgger import Swagger
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import base64
import logging
from pymongo import MongoClient
import datetime
import time
import threading

app = Flask(__name__)
CORS(app)

# set up logging
logging.basicConfig(level=logging.INFO)

# load the model and tokenizer
model_path = '/home/ec2-user/scam_detector_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# MongoDB connection
uri = "mongodb+srv://coen424team3u:e23zM9AnYHjcGiQg@cluster0.8nrxy.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"
client = MongoClient(uri)
db = client['spam_detect']
input_messages = db['messages']
db.messages.create_index("message_id", unique=True)
output_predictions = db['predictions']

# initialize Swagger
swagger = Swagger(app)

# for swaggerhub expose endpoint classify email
@app.route('/classify_email', methods=['POST'])
def classify_email_endpoint():
    """
    Classifies the email content as spam or not.
    ---
    parameters:
      - name: email_body
        in: body
        required: true
        type: string
        description: The email content to classify as spam or not spam.
    responses:
      200:
        description: The classification result (Spam or Not Spam) with confidence score.
        schema:
          type: object
          properties:
            prediction:
              type: string
              enum: [Spam, Not Spam]
            confidence_score:
              type: number
              format: float
      400:
        description: Bad Request. Invalid input.
    """
    # Extract email content from the request body
    try:
        data = request.json
        email_content = data.get('email_body')
    except Exception:
        return jsonify({"error": "Invalid JSON or no input provided"}), 400

    # make sure input exists
    if not email_content:
        return jsonify({"error": "The 'email_body' field is required."}), 400

    # classify
    prediction, confidence_score = classify_email(email_content)

    # return the classification result
    return jsonify({
        "prediction": prediction,
        "confidence_score": confidence_score
    }), 200

@app.route('/authorize')
def authorize():
    """
    Initiates the OAuth2 flow for Gmail API authorization.
    ---
    responses:
      302:
        description: Redirects to Google's authorization page.
    """
    flow = Flow.from_client_secrets_file( # create OAuth2 flow object given google api credentials.json
        '/home/ec2-user/credentials.json',
        scopes=['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify'], # permissions requested
        redirect_uri='https://ec2-3-146-122-143.us-east-2.compute.amazonaws.com/oauth2callback' # redirect
    )
    authorization_url, _ = flow.authorization_url(access_type='offline', include_granted_scopes='true') # offline for even if the user is not logged in
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    try:
        flow = Flow.from_client_secrets_file(
            '/home/ec2-user/credentials.json',
            scopes=['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify'],
            redirect_uri='https://ec2-3-146-122-143.us-east-2.compute.amazonaws.com/oauth2callback'
        )

        https_authorization_url = request.url.replace('http://', 'https://') # because gmail api requires https
        flow.fetch_token(authorization_response=https_authorization_url)
        credentials = flow.credentials
        app.logger.info("OAuth2 credentials successfully fetched.")

        service = build('gmail', 'v1', credentials=credentials) # create gmail api service object to interact with gmail api
        
        # periodic email checking in separate thread
        threading.Thread(target=periodic_new_email_check, args=(service,), daemon=True).start()

        return "You have successfully authorized the application!"

    except Exception as e:
        app.logger.error(f"Error during OAuth2 callback: {str(e)}")
        return f"An error occurred during authorization: {str(e)}", 500

def periodic_new_email_check(service):
    while True:
        try:
            # fetch new unread emails
            app.logger.info("Checking for new emails...")
            process_emails(service, query='is:unread')
        except Exception as e:
            app.logger.error(f"Error while polling emails: {e}")

        # check for new unread emails every 5 seconds
        time.sleep(5)

def process_emails(service, query='is:unread'):
    # fetch unread emails
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], q=query).execute()
    messages = results.get('messages', [])

    if not messages:
        app.logger.info("No new emails found.")
        return

    # loop through messages and process them
    for msg in messages:
        msg_id = msg['id']
        app.logger.info(f"Processing email {msg_id}...")

        # check if the email id already exists in MongoDB. If yes then skip
        if input_messages.find_one({"message_id": msg_id}):
            continue

        # Fetch the email content
        msg_data = service.users().messages().get(userId='me', id=msg_id).execute()
        email_data = msg_data['snippet']

        # skip if the email starts with "Your email is classified as" since it means that it has already been classified
        if email_data.startswith("Your email is classified as"):
            app.logger.info(f"Email {msg_id} has already been handled. Skipping...")
            continue

        # classify email as spam or not spam
        prediction, confidence_score = classify_email(email_data)

        # save to MongoDB
        app.logger.info(f"Saving email {msg_id} to MongoDB.")
        message_data = {
            'message_id': msg_id,
            'content': email_data,
            'timestamp': datetime.datetime.now(datetime.timezone.utc),
        }
        input_messages.insert_one(message_data)

        # save classification result to MongoDB
        app.logger.info(f"Saving classification result for email {msg_id} to MongoDB.")
        result = {
            'message_id': msg_id,
            'prediction': prediction,
            'confidence_score': confidence_score,
            'timestamp': datetime.datetime.now(datetime.timezone.utc),
        }
        output_predictions.insert_one(result)

        send_reply(service, prediction, msg_data)

# classify email as spam or not spam
def classify_email(email_content):
    inputs = tokenizer(email_content, return_tensors="pt", truncation=True, padding=True) # tokenize

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    confidence_score = torch.max(outputs.logits.softmax(dim=1)).item()

    return ('Spam' if prediction == 1 else 'Not Spam', confidence_score)

# send a reply to the email indicating if it is classified as spam or not.
def send_reply(service, prediction, msg_data):
    msg_id = msg_data['id']

    recipient_email = "coen424team3U@gmail.com"

    # reply
    email_reply = f"To: {recipient_email}\nSubject: Re: Spam Classification Result\n\nYour email is classified as: {prediction}"

    # encode message
    raw_message = base64.urlsafe_b64encode(email_reply.encode("utf-8")).decode("utf-8")
    message = {
        'raw': raw_message,
        'threadId': msg_data.get('threadId')
    }

    try:
        # send reply
        service.users().messages().send(userId='me', body=message).execute()
        app.logger.info(f"Reply sent to {recipient_email} regarding spam status: {prediction}.")

    except Exception as e:
        app.logger.error(f"Error sending reply for email {msg_id}: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
