import requests

slack_webhook_url = "put your link"

def send_message_slack(text: str) -> None:
    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text,
                },
            },
        ],
    }
    # requests.post (WebhookUrl. SLACK_WEBHOOK_URL, json=payload)

    requests.post (slack_webhook_url, json=payload)

if __name__ == "__main__":
    send_message_slack(text="Model load completed")