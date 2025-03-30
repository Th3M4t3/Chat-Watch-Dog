import requests

class LibreTranslateClient:
    def __init__(self, host='localhost', port=5000):
        self.url = f'http://{host}:{port}/translate'
        if not self._is_libretranslate_running(host, port):
            raise ConnectionError(f"LibreTranslate is not running on {host}:{port}")
        else:
            print(f"LibreTranslate is running on {host}:{port}")

    def _is_libretranslate_running(self, host, port):
        try:
            response = requests.get(f'http://{host}:{port}/')
            # Check if the response is successful
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def translate(self, text, source_lang='auto', target_lang='en'):
        data = {
            'q': text,
            'source': source_lang,
            'target': target_lang
        }
        
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                translated_text = response.json().get('translatedText', '')
                return translated_text
            else:
                raise Exception(f"Error in translation: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"An error occurred: {e}")
