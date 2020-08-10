from nn import get_generators, get_model, train_model

train_path = 'train'
val_path = 'validation'
img_dim = (256, 256)

# creating data generators
train_generator, val_generator = get_generators(train_path, val_path, img_dim)

input_dim = (256, 256, 3)

model = get_model(input_dim)
history = train_model(model, train_generator, val_generator)

loss, acc = model.evaluate(val_generator, verbose=2)

print(f'Training Loss: {loss}')
print(f'Model Accuracy: {acc}')

# saves the model if the accuracy is greater than 80%
if acc > 80:
    model.save('model.h5')