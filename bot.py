import discord
from responses import handle_response
from huggingface import NLPModel
import csv
from datetime import datetime
import json
from translator import LibreTranslateClient



def write_to_csv(filename, username, message, channel, date, model_output, is_grooming):
	with open(filename, 'a', newline='') as csvfile:
		fieldnames = ['Username', 'Message', 'Channel', 'Date', 'Model_output', 'Is_grooming']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writerow({'Username': username, 'Message': message, 'Channel': channel, 'Date': date,'Model_output': model_output, 'Is_grooming': is_grooming})

def run_discord_bot():
	with open('config.json') as f:
		config = json.load(f)

	discord_config = config['discord']
	libre_config = config['libretranslate']
	host = libre_config['host']
	port = libre_config['port']
	token = discord_config['API_KEY']
	lookback = discord_config['Lookback']

	treshold = config['grooming_treshold']
	client = discord.Client(intents = discord.Intents.all())
	model = NLPModel()
	translator = LibreTranslateClient(host, port)

	@client.event
	async def on_ready():
		print(f'{client.user} is now running!')

	@client.event
	async def on_message(message):
		if message.author == client.user:
			return

		username = str(message.author)
		user_message = message.content
		date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		
		try:
			messages = []
			async for msg in message.channel.history(limit=lookback):
				messages.append(msg.content)
			messages.reverse()
			translated_messages = []
			for msg in messages:
				translated_messages.append(translator.translate(msg, "auto", "en"))
			for msg in translated_messages:
				print(msg)
			model_output = model.run_inference_model(translated_messages)
			is_grooming = []
			for out in model_output:
				if out > treshold:
					is_grooming.append(True)
					response = handle_response(out)
					await message.channel.send(response)
				else:
					is_grooming.append(False)
			print(model_output)
			write_to_csv('./discord_data.csv', username, user_message, str(message.channel), date, model_output, is_grooming)
		
		except discord.Forbidden as e:
			print(f"Permission error: {e}")
			return
		except Exception as e:
			print(f"An error occurred: {e}")
			return
		except ConnectionError as e:
			print(f"An error occured when connecting: {e}")
			return

	client.run(token)