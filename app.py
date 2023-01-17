import os
import random
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from numpy import newaxis, argmax

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the EMO flask api      -[dev. by Raj Kishore]:)"


@app.route("/predict", methods=['POST', 'OPTIONS'])
def predict():
    audioFile = request.files['file']
    audioFileName = str(random.randint(0, 1000))
    audioFile.save(audioFileName)

    model = tf.keras.models.load_model("EMOmodel.h5")

    input_data, input_sr = librosa.load(audioFileName, sr=22050)

    codes = ["angry", "sad", "happy", "surprised", "neutral", "disgust", "fear", "calm"]
    sec_2 = 2 * input_sr

    n = int(len(input_data) / sec_2)
    T = 0
    ans = []

    for i in range(n):

        signal = input_data[i * sec_2: (i + 1) * sec_2]

        input_mfccs = librosa.feature.mfcc(signal, input_sr, n_mfcc=45, n_fft=2048, hop_length=512)
        input_mfccs = input_mfccs[newaxis, ..., newaxis]
        input_mfccs = input_mfccs.T

        j = argmax(model.predict(input_mfccs)[0])

        T = T + 2

        if len(ans) == 0 or ans[-1][2] != codes[j]:

            minu, sec = divmod(T, 60)
            hr, minu = divmod(minu, 60)
            t = f"{hr}:{minu}:{sec}"

            if len(ans) == 0:
                ans.append(["0:0:0", "0:0:2", codes[j]])
            elif len(ans) == 1:
                ans[0][1] = t
            else:
                ans.append([ans[-1][1], t, codes[j]])

    os.remove(audioFileName)

    if len(ans) == 0:
        ans.append([0, 0, "Null"])

    return jsonify(ans)


if __name__ == "__main__":
    app.run(debug=True)
