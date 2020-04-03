from imports import *

f = open('results.txt', 'w')

def fourier_prediction(complexPart):
    
    """[Função principal, vai fazer a previsão do KNN, 
        pegando partes da transformada de fourier (real e/ou imaginaria]
    
    Arguments:
        complexPart {[string]} -- [ Podem ser:
                                    - real 
                                    - imaginary
                                    - sum
                                    - real_and_imaginary
                                    - merge]
    """
    scores = []
    
    for img_size in range(2, 51):
    
        train_imgs = []
        test_imgs = []
        y_pred = []

        #? Aqui são inserido em uma lista as imagens re-dimensionadas de teste
        for test in fft_centered_test:
            test_imgs.append(np.array(smallest_subregion(img_size, test)).flatten()) 
        
        #? Aqui são inserido em uma lista as imagens re-dimensionadas de treino
        for train in fft_centered_train:
            train_imgs.append(np.array(smallest_subregion(img_size, train)).flatten())
        
        #? Aqui a lista de treino é transformada em array para poder ser parametro da função "fit"
        x_train = np.array(train_imgs)
        
        #? Aqui é criado o modelo a ser utilizado para a identificação de padrões utilizando knn com um vizinho
        #? para imagens com esta dimensão
        model = KNeighborsClassifier(n_neighbors=1, metric="euclidean")

        #? Aqui é realizado o treinamento do modelo considerando apenas a parte real
        if complexPart == 'real':
            model.fit(x_train.real, y_train)
            
            for test in test_imgs:
                y_pred.append(model.predict([test.real]))
        
        elif complexPart == 'imaginary':
            
            model.fit(x_train.imag, y_train)
            
            for test in test_imgs:
                y_pred.append(model.predict([test.imag]))
                
        elif complexPart == 'sum':
            
            model.fit(x_train.real + x_train.imag, y_train)
            
            for test in test_imgs:
                y_pred.append(model.predict([test.real + test.imag]))

        elif complexPart == 'real_and_imaginary':
            #? Aqui sera feito a chamada passando apenas a parte Real e/ou Imaginária para análise
            model_real = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
            model_imag = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
            model_real_imag = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
            model_imag_real = KNeighborsClassifier(n_neighbors=1, metric="euclidean")

            model_real.fit(x_train.real + x_train.imag, y_train)
            model_imag.fit(x_train.real + x_train.imag, y_train)
            model_real_imag.fit(x_train.real + x_train.imag, y_train)
            model_imag_real.fit(x_train.real + x_train.imag, y_train)
            
            for test in test_imgs:
                y_pred_real=(model_real.predict([test.real]))
                y_pred_imag=(model_imag.predict([test.imag]))
                y_pred_real_imag=(model_real_imag.predict([test.imag]))
                y_pred_imag_real=(model_imag_real.predict([test.real]))
            
                least_dist = [y_pred_real, y_pred_imag, y_pred_real_imag, y_pred_imag_real]
                least_dist_sorted = sorted(least_dist)[0]
                y_pred.append(least_dist_sorted)

        elif complexPart == 'merge':
            
            x_train_abs = abs(x_train)
            model.fit(x_train_abs, y_train)
            
            for test in test_imgs:
                y_pred.append(model.predict([abs(test.real) + abs(test.imag)]))
        
        y_pred = np.array(y_pred) 
        score = accuracy_score(y_true, y_pred)
        scores.append(score)
        
        f.write(f'Utilizando: {complexPart}\n')

        f.write(f'Taxa de acerto com a foto {img_size}x{img_size}: {score*100}%\n')

        f.write(f'Taxa de Erro {img_size}x{img_size}: {100 - score*100}%\n\n')

    f.write(f'Score medio do tipo [{complexPart}]: {np.mean(score)*100}%\n')
    f.write(f'\n----------------------------------------------------------\n\n')


def smallest_subregion(diameter, image):
    #? Pegando os valores centrais da imagem após recentralizá-los
    r = diameter//2
    #? Altura considerando somente a parte inteira
    x = width//2 
    #? Largura considerando apenas a parte inteira
    y = height//2 
    
    odd = 0 
    if diameter % 2 != 0:
        odd = -1 
    
    return image[x - r + odd: x + r, y - r + odd: y + r]

fft_train, fft_centered_train = [], []
fft_test, fft_centered_test = [], []
label_train, label_test = [], []
people_test, people_train = [], []
y_train, y_true = [], []

#? Passa por todas as imagens do banco de dados cedido e
#? Cria novas imagens utilizando transformada de fourier bidimensional
for s_value in range(1, 41):
    
    y_true.append(s_value)
    random_image_num = random.randint(1, 10)

    for img_value in range(1, 11):
        
        img = Image.open(f'orl_faces/s{s_value}/{img_value}.pgm')

        img_array = np.asarray(img)

        #? aplicada a transformada de fourier bidimentsional
        fft_arr = np.fft.fft2(img_array)
        
        #? coloca no treino se for a imagem aleatória que pegou
        if img_value == random_image_num:
            fft_test.append(fft_arr)
            fft_centered_test.append(np.fft.fftshift(fft_arr))
            label_test.append(s_value)
            people_test.append(img)

        #? se não for, coloca em teste
        else:
            
            fft_train.append(fft_arr)
            fft_centered_train.append(np.fft.fftshift(fft_arr))
            label_train.append(s_value)
            people_train.append(img)
            y_train.append(s_value)
            
            
width, height = np.array(people_test[0].size)


fourier_prediction('merge')
fourier_prediction('real')
fourier_prediction('imaginary')
fourier_prediction('sum')
fourier_prediction('real_and_imaginary')

f.close()


