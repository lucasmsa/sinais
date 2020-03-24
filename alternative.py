from imports import *

# redimensiona e achata uma imagem transformando-a 
# em uma lista de intensidade de pixels
def image_to_feature_vector(image, size=(10, 10)):
        
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
        
        # aplicada a transformada de fourier bidimentsional
        fft_arr = abs(np.fft.rfft2(img_array))
        
        new_img = Image.fromarray(fft_arr)

        if new_img.mode != 'RGB':
            new_img = new_img.convert('RGB')

        new_img.save(f'fft_faces/s{s_value}_{img_value}.png')

        counter += 1
        
#print(fft_arr)

image_paths = list(paths.list_images('fft_faces'))

# para cada imagem no banco de dados preencher as 
# listas supracitadas com os valores determinados

# intensidade dos pixels
pixelIntensitiesBid = [[] for i in range(40)]
# histograma
histFeaturesBid = [[] for i in range(40)]
# uma das 41 pessoas do dataset
classLabelsBid = [[] for i in range(40)]

count = 0
person = 0

for (i, image_path) in enumerate(sorted(image_paths)):

    image = cv2.imread(image_path)
    # pegar a pessoa da imagem pelo nome do arquivo
    label = image_path.split(os.path.sep)[-1].split('_')[0]
    label_numeric = label.split('s')[1]
    # pegar valores de intensidade do pixel, depois o histograma
    # obtendo a distribuicao dos tons de preto na imagem
    pixels = image_to_feature_vector(image)
    histogram = extract_grayscale_histogram(image)

    pixelIntensitiesBid[person].append(pixels)
    histFeaturesBid[person].append(histogram)
    classLabelsBid[person].append(int(label_numeric))

    count += 1

    if count % 10 == 0:
        person += 1

pixelIntensitiesNew = []
pixelIntensitiesNew = np.array(pixelIntensitiesNew)

# ANCHOR * 
for i in range(40):
    # dividir os dados em treino e teste, 90% treino 10%
    # a priori, utilizando a intensidade dos pixels
    (train_PI, test_PI, train_L, test_L) = train_test_split(
    pixelIntensitiesBid[i], classLabelsBid[i], test_size=0.1, random_state=42)

    print(f'\n{train_PI}')

    model = KNeighborsClassifier(n_neighbors=1)
    # treina o modelo atraves desse .fit
    model.fit(train_PI, train_L)
    accuracy = model.score(test_PI, test_L)
    model_accuracy = accuracy*100
    print(f'{i} - Model accuracy Pixel Intensities: {model_accuracy}')


# TODO: Achar uma maneira de juntar esses valores das imagens separadas para treino e teste no for marcado pela anchor tag
# TODO  - para depois serem usados no treinamento do KNN no final

# TODO: Descobrir como fazer o MSE, tentei pela funcao mas do sklearn mas deu erro
# TODO  - talvez esse link ajude: https://machinelearningmastery.com/implement-machine-learning-algorithm-performance-metrics-scratch-python/
# ? Problema eh que nao sei exatamente o que deve ir para os parametros do mean squared error

# TODO: Rodar os testes que ele pediu, acredito que essa parte nao seja tao dificil,
# TODO - o que gera a transformada de fourier bidimensional eh essa parte do codigo 
# TODO -- fft_arr = abs(np.fft.rfft2(img_array)), eh soh para cada imagem realizar os testes que ele pede, *acho*
# TODO --- https://stackoverflow.com/questions/43001729/how-should-i-interpret-the-output-of-numpy-fft-rfft2

# TODO: A parte de achar a sub-regiao com menor frequencia tambem nao sei se esta sendo feita 
# TODO - acredito que isso so vai requerer mudancas na funcao image_to_feature_vector, ela ja tem um parametro size
# TODO -- mas acho que ele so redimensiona a imagem, e nao trata de achar essa sub-regiao, ver esse link: https://www.freecodecamp.org/news/getting-started-with-tesseract-part-ii-f7f9a0899b3f/
# TODO --- talvez esse tambem ajude: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html, ou esse: https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
