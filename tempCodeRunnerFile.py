from imports import *

def image_to_feature_vector(image, size=(30, 30)):
    # redimensiona e achata uma imagem transformando-a 
    # em uma lista de intensidade de pixels    
    return cv2.resize(image, size).flatten()


# construir o histograma da imagem para caracterizar a distribuicao de tons de preto desta
def extract_grayscale_histogram(image):
    # extrai os tons de cinza
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # para uma imagem em preto e branco 8-bits de informacao por pixel, existem 256 diferentes possibilidades 
    # de intensidade, assim, o histograma tem 256 valores
    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])

    # normalizacao do histograma
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


try:
    os.makedirs('fft_faces')
except Exception:
    pass

counter = 0
# passa por todas as imagens do banco de dados cedido e
# cria novas imagens utilizando transformada de fourier bidimensional
for s_value in range(1, 41):

    for img_value in range(1, 11):

        img = Image.open(f'orl_faces/s{s_value}/{img_value}.pgm')
        
        img_array = np.asarray(img)
        
        fft_arr = abs(np.fft.rfft2(img_array))

        new_img = Image.fromarray(fft_arr)

        if new_img.mode != 'RGB':
            new_img = new_img.convert('RGB')

        new_img.save(f'fft_faces/s{s_value}_{counter}.png')

        counter += 1
        

image_paths = list(paths.list_images('fft_faces'))

# intensidade dos pixels
pixelIntensities = []
# histograma
histFeatures = []
# uma das 41 pessoas do dataset
classLabels = []

# para cada imagem no banco de dados preencher as 
# listas supracitadas com os valores determinados
for (i, image_path) in enumerate(image_paths):
    
    image = cv2.imread(image_path)
    # pegar a pessoa da imagem pelo nome do arquivo
    label = image_path.split(os.path.sep)[-1].split('_')[0]
    
    # pegar valores de intensidade do pixel, depois o histograma
    # obtendo a distribuicao dos tons de preto na imagem
    pixels = image_to_feature_vector(image)
    histogram = extract_grayscale_histogram(image)

    pixelIntensities.append(pixels)
    histFeatures.append(histogram)
    classLabels.append(label)

    
# dividir os dados em treino e teste, 90% treino 10%
# a priori, utilizando a intensidade dos pixels
(train_PI, test_PI, train_L, test_L) = train_test_split(
    pixelIntensities, classLabels, test_size=0.1, random_state=42
)
# utiliza-se aqui o histogrma dos tons de preto para treino 
# e teste
(train_HF, test_HF, train_HF_L, test_HF_L) = train_test_split(
    histFeatures, classLabels, test_size=0.1, random_state=42
)

# treinamento do KNN com 1 vizinho, utilizando o Pixel Intensities
model = KNeighborsClassifier(n_neighbors=1)
model.fit(train_PI, train_L)
accuracy = model.score(test_PI, test_L)
model_accuracy = accuracy*100
print(f'Model accuracy: {model_accuracy}')