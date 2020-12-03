import spacy

from audio_parser import AudioParser


class SentimentAnalysis:
    def __init__(self):
        self.audio_parser = AudioParser()

    def getSentiment(self):

        response = self.audio_parser.recognize_speech_from_mic()
        if response['success']:
            input_data = response['transcription']
        else:
            raise response['error']

        #  Load saved trained model
        loaded_model = spacy.load("./model_artifacts")
        parsed_text = loaded_model(input_data)

        # Determine prediction to return
        if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
            prediction = "Positive"
            score = parsed_text.cats["pos"]
        else:
            prediction = "Negative"
            score = parsed_text.cats["neg"]
        return prediction, score


if __name__ == "__main__":
    sa = SentimentAnalysis()
    pred, scr = sa.getSentiment()
