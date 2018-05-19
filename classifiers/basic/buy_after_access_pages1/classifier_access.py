# minha abordagem inicial foi
# 1. separar 90% para treino e 10% para teste: 88.89%

from page_flow import low_page_access

X,Y = low_page_access()
data_train = X[:90]
target_train = Y[:90]

data_test = X[-9:]
target_test = Y[-9:]

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(data_train, target_train)

result_from_model = model.predict(data_test)

print(target_test)
print(result_from_model)


deltas = result_from_model - target_test

hits = [d for d in deltas if d == 0]
hits_total = len(hits)
number_of_test = len(data_test)
hits_percent = 100.0 * hits_total / number_of_test

print(hits_percent)
print(number_of_test)