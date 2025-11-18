from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score




def train_classifier(X_train, y_train, X_test, y_test):
model = MultinomialNB()
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
return model, accuracy
