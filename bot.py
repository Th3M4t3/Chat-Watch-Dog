import discord
import responses
from huggingface import bert_model
import csv
from datetime import datetime
import json



def write_to_csv(filename, username, message, channel, date, is_grooming):
	with open(filename, 'a', newline='') as csvfile:
		fieldnames = ['Username', 'Message', 'Channel', 'Date', 'Is_grooming']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writerow({'Username': username, 'Message': message, 'Channel': channel, 'Date': date, 'Is_grooming': is_grooming})

def run_discord_bot():
	with open('config.json') as f:
		config = json.load(f)

	TOKEN = config['API_KEY']
	client = discord.Client(intents = discord.Intents.all())
	model = bert_model()

	@client.event
	async def on_ready():
		print(f'{client.user} is now running!')

	@client.event
	async def on_message(message):
		if message.author == client.user:
			return  # Prevent bot from responding to itself
	
		# Lookback limit
		lookback = 4
		
		# Extract relevant message info
		username = str(message.author)
		user_message = message.content
		date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		
		try:
			# Fetch the last 'lookback' number of messages and extract only their content
			messages = []
			async for msg in message.channel.history(limit=lookback):
				messages.append(msg.content)
			messages.reverse()
			for msg in messages:
				print(msg)
			# Run inference model with the message contents (texts)
			is_grooming = model.run_inference_model(messages)
			print(is_grooming)
			
			# Save the data to CSV
			write_to_csv('../dataset/discord_data.csv', username, user_message, str(message.channel), date, is_grooming)
		
		except discord.Forbidden as e:
			print(f"Permission error: {e}")
			return
		except Exception as e:
			print(f"An error occurred: {e}")
			return

	client.run(TOKEN)