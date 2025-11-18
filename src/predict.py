def predict_text(model, vectorizer, text):
vec = vectorizer.transform([text])
return model.predict(vec)[0]
