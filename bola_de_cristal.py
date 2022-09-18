#este proyecto fue diseñado con fines educativos en ningun momento es un consejo de inversion 
#la version original del proyecto esta diseñada en google colaboratory devido a los requerimientos de hardware 
#el proyecto hace uso de elementos de inteligencia artificial para desarrollar un modelo de macine learning que prediga 
#el precio de el mercado de acciones 



#importamos las librerias con las que trabajaremos
import pandas as pd 
import numpy as np
import matplotlib.pylab as plt 

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers  import Dense, Activation, Flatten
#from keras.layers import Dense, LSTM, Dropout

#esto es para la version de google colaboratory y 
#es un archivo de excel de donde importaremos los datos 
from google.colab import files 
uploaded = files.upload()

#generamos el dataframe de trabajo 
df = pd.read_csv('time_series.csv', parse_dates=[0], header=None, index_col=0, squeeze=True, names=['fecha','unidades'])
df.head()

#obtenemos los datos generales de la tabla 
df.describe()

#obtenemos el dato maximo y minimo del data frame 
print(df.index.min())
print(df.index.max())

#clasificamos la informacion por meses  obtenemos su valor promedio y la imprimimos 
meses = df.resample('M').mean()
meses

#graficamos la informacion de los meses asi como su promedio  
plt.plot(meses['2017'].values)
plt.plot(meses['2018'].values)


#establecemos el numero de pasos de nuestra red neuronal
PASOS=7

#convertimos la informacion de  series a aprendizaje supervisado 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  df = pd.DataFrame(data)
  cols, names= list(), list()

  #secuencia de entrada(t-n,.. t+n)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
      
  #forecast secuencia (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for  j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
      
  #ponemos tdo junto
  agg = pd.concat(cols, axis=1)
  agg.columns = names

  #ponemos las filas con valores Nan para evitar errores
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#cargamos el dataset
values = df.values

#nos aeguramos que todos os datso sean float 
values = values.astype('float32')

#normalizamos caracteristicas 
scaler = MinMaxScaler(feature_range=(-1,1))
values = values.reshape(-1,1) # se realiza este paso debido a que solo poseemos una dimension
scaled = scaler.fit_transform(values)

# frame como aprendizaje supervisado(supervised_learning)
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.head()


 # dividimos en set de entrenamiento y validacion 
 values = reframed.values 
 n_train_days = 315 + 289 - (30+PASOS)
 train = values[:n_train_days, :]
 test = values[n_train_days:, :]

 #dividimos en entradas y salidas 
 x_train, y_train  = train[:, :-1],train[:,-1]
 x_val, y_val = test[:, :-1], test[:, -1]

 #redimensionamos a una matriz 3d[samples, timesteps, features]
 x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
 x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
 
 print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

  
#--------------------------------creamos el modelo de la red neuronal--------------------------------------------------------- 
#----------usamos una red "normal"  Feedfordward FF
def crear_modeloFF():
  model=Sequential()
  model.add(Dense(PASOS, input_shape=(1, PASOS), activation='tanh'))
  model.add(Flatten())
  model.add(Dense(1, activation='tanh'))#la funcion de activaciond e la red sera la tanhipervolica debido a su ligereza de procesamiento
  model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=["mse"])
  model.summary()
  return model


#--------------------------------------entrenamiento de la maquina------------------------------------------ 
EPOCHS=40     #epocas de entrenamiento
model = crear_modeloFF()     
history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=PASOS)

#--------------------------------------ploteamos los resultados obtnidos del modelo y su prediccion-----------------

results = model.predict(x_val)
print( len(results) )
#plt.scatter(range(len(y_val)), y_val, c='g')
plt.plot(range(len(y_val)), y_val, c='g')
#plt.scatter(range(len(results)), results, c='r')
plt.plot(range(len(results)), results, c='r')
plt.legend(['validacion', 'resultados'])
plt.title('validate')
plt.show()            


#con una grafica de puntos y la validacion del resultado podemos observar la presicion del modelo asi como tambien podemos filtrar los datos 
#podemos mejorar la presicion del modelo haciendo uso de la regresion lineal 
print( len(results) )
plt.scatter(range(len(y_val)), y_val, c='g')
 
plt.scatter(range(len(results)), results, c='r')

plt.legend(['validacion', 'resultados'])
plt.title('validate')
plt.show()

#de forma mas especifia podemos obsrvar el grado de error de nuestro modelo
plt.plot(history.history['loss'])
plt.title('loss')
plt.plot(history.history['val_loss'])
plt.title('validate loss')
plt.show()


#esta grafica nos muestra como la presicion del modelo decae cuanto mas se aleja de la muetra inicial 
#este decadencia en la presicion es devido a la aleatoriedad de los valores del mercado 
#tmabien se puede observar como la presicion decae fuertemente hasta un punto de estabilizacion
plt.title('Accuracy')
plt.plot(history.history['mse'])
plt.show()  

#esta es una forma mas literal de describir el comportamiento del modelo 
compara = pd.DataFrame(np.array([y_val, [x[0] for x in results]])).transpose()
compara.columns = ['real','prediction']

inverted = scaler.inverse_transform(compara.values)

compara2 = pd.DataFrame(inverted)
compara2.columns = ['real','prediction']
compara2['diferencia'] = compara2['real']-compara2['prediction']
compara2.head()

#visualizacion de los valores reales y prediccionde en los vlaores de todas las columnas del dataframe
compara2.describe()

#aqui se pone a prueba el grado de preciosion del modelo e intenta predecir datos que no estan dentro del dataset de trabajo 

#-----------------------------------------PREDICCION DE LOS DATOS A FUTURO-------------------------------------------------- 
#aparti de la primera semana de noviembre de 2018 intentaremos predecir la primera semana de diciembre 
ultimosDias = df['2018-11-16':'2018-11-30']
ultimosDias

#preparmos los datos para el test 
values = ultimosDias.values
values = values.astype('float32')

#normalizamos las caracteristicas 
values = values.reshape(-1, 1)    #hacemos esto por que tenemos 1 soa dimension 
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.drop(reframed.columns[[7]], axis=1, inplace=True)
reframed.head(7)

values = reframed.values
x_test = values[6:, :]
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_test.shape)
x_test

#utlizamos la nueva funcion creada para predecir los nuevos valores
# tenemos que reconvertir los datos de la prediccion a los datos legibles para el usuario
#  
def agregarNuevoValor(x_test,nuevoValor):
  for i in range(x_test.shape[2]-1):
    x_test[0][0][i] = x_test[0][0][i+1]
  x_test[0][0][x_test.shape[2]-1]= nuevoValor
  return x_test 

  #pronostico a futuro 
results = []
for i in range (7):
  parcial = model.predict(x_test)
  results.append(parcial[0])
  print(x_test)
  x_test = agregarNuevoValor(x_test, parcial[0])


#RECONVERTIMOS LOS RESULTADOS 
adimen = [x for x in results]
print(adimen)
inverted = scaler.inverse_transform(adimen)
inverted


#VISUALIZACION LA PREDICCION GENERADO POR EL PRONOSTICO 
prediccion1SemanaDiciembre = pd.DataFrame(inverted)
prediccion1SemanaDiciembre.columns = ['pronostico']
prediccion1SemanaDiciembre.plot()
prediccion1SemanaDiciembre.to_csv('pronostico.csv')

#IMPRIMIMOS LOS DATOS 
prediccion1SemanaDiciembre

#agregamos el resultado al dataset 
i=0
for fila in prediccion1SemanaDiciembre.pronostico:
  i=i+1
  ultimosDias.loc['2018-12-01'+str(i)+'00:00:00'] = fila
  print(fila)

ultimosDias.tail(14)