import numpy as np
import matplotlib.pyplot as plt

def nodesForGrid(sizeCageX, formCages):
    numberCages = [i for i in range(1, sizeCageX ** 2 + 1)]     #   [1, 2, 3, 4, 5, 6, 7, 8, 9]
    counterNodes = numberCages[0] + sizeCageX                   #   4
    startNumberNodes = numberCages[0] + 1                       #   1
    cloudAllNodes = []

    ### Генерация узлов для каждой ячейки сетки
    if formCages == 'rectangle':
        numberNodesForCage = 4
        paramsForNodes = [counterNodes, numberCages[0], (-1.) * counterNodes, (-1.) * numberCages[0]]
        cloudNodesForCage = [0. for i in range(numberNodesForCage)]
        cloudAllNodes = [[0. for i in range(numberNodesForCage)] for j in range(len(numberCages))]

        for i in range(len(cloudAllNodes)):
            for j in range(len(cloudAllNodes[0])):
                cloudAllNodes[i][j] = j + i

        for i in numberCages:
            for j in range(numberNodesForCage):
                if j == 0:
                    cloudNodesForCage[j] = (startNumberNodes - 1) + paramsForNodes[j]
                    cloudAllNodes[i - 1][j] = cloudNodesForCage[j]
                    continue
                cloudNodesForCage[j] = cloudNodesForCage[j - 1] + paramsForNodes[j]
                cloudAllNodes[i - 1][j] = cloudNodesForCage[j]
            startNumberNodes += 1
            if i % sizeCageX == 0: startNumberNodes += 1
    return cloudAllNodes

def getValuesForNodes(hx, hy, gradus, sizeCageX, formCages):
    # Не полностью реализовал размер сетки, больше 5
    maxSizePointX = -(sizeCageX * hx / 2.)
    maxSizePointY = sizeCageX * hy / 2.
    deltaX = np.cos(np.math.pi / 180. * (gradus)) * hy
    deltaY = np.sin(np.math.pi / 180. * gradus)
    xAxle, yAxle, cloudForAllNodes = [], [], []
    listCoefficientsForCageX = [0. for i in range(sizeCageX + 1)]

    ### Генерация координат для каждого узла сетки
    ### В начале создаю главную линию по X Y, создавая тем самым псевдограницы для ячеек сетки
    for i in range(sizeCageX):
        if i == 0:
            xAxle.append(maxSizePointX)
            yAxle.append(maxSizePointY)
        maxSizePointX = maxSizePointX + hx
        maxSizePointY = maxSizePointY - hy
        xAxle.append(maxSizePointX)
        yAxle.append(maxSizePointY)

    ### Если угол ячеек не равен 90 градусов, то начинаю смещать каждую
    ### прямоугольную ячейку, превращая её в параллелограмм. Здесь преобразую X компоненту
    for i in range(int((sizeCageX + 1) / 2) - 1, -1, -1):
        if i == (sizeCageX + 1) / 2 - 1:    listCoefficientsForCageX[i] = deltaX / 2.
        else:                               listCoefficientsForCageX[i] = listCoefficientsForCageX[i + 1] + deltaX
    for i in range(int((sizeCageX + 1) / 2), (sizeCageX + 1), 1):
        if i == (sizeCageX + 1) / 2:    listCoefficientsForCageX[i] = -(deltaX) / 2.
        else:                           listCoefficientsForCageX[i] = listCoefficientsForCageX[i - 1] - deltaX

    ### Заполняю массив преобразованными X и Y, но в начале закладываю в него центральную точку.
    ### Делаю это для того,чтобы не пришлось менять нумерацию узлов в других частях программы,
    ### т.к. нумерация всех массивов начинается с единицы, что неудобно для моего алгоритма.
    cloudForAllNodes.append([0., 0.])
    for i in range(len(yAxle)):
        for j in range(len(xAxle)):
            cloudForAllNodes.append([xAxle[j] + listCoefficientsForCageX[i], yAxle[i] * deltaY])
    return cloudForAllNodes

def getTraceOfMatrix(sizeCageX, cloudAllNodes, formCages):
    # Не реализовал размер сетки, больше 5
    quantity, counter = 0, 0
    if formCages == 'rectangle': quantity = 4
    listForTraceOfMatrix = [[[0. for z in range(quantity)] for j in range(sizeCageX)] for i in range(sizeCageX)]

    ### Создаю слепок матрицы в виде трёхмерного массива, т.е. буквально копирую вид сетки 3х3
    ### и переношу это изображение в массив.
    for i in range(sizeCageX):
        for j in range(sizeCageX):
            for z in range(quantity):
                listForTraceOfMatrix[i][j][z] = cloudAllNodes[counter][z]
            counter += 1
    return listForTraceOfMatrix

def countXYForGraficsXY_PNG(sizeCageX, hx, hy, gradusBetta, coordinateForNodes, cloudAllNodes, formCages):
    ### Расчёт градиента для угла от 90 до 1 градуса. Отрисовка и сетки, и XY компонент в соответсвии с изменением угла
    cloudGradX, cloudGradY, formPicture, cloudForGraduent = [], [], 0., []
    while gradusBetta != 1.:
        coordinateForNodes = getValuesForNodes(hx, hy, gradusBetta, sizeCageX, formCages)
        cloudAllNodes = nodesForGrid(sizeCageX, formCages)
        listForTraceOfMatrix = getTraceOfMatrix(sizeCageX, cloudAllNodes, formCages)
        cloudForGraduent = getResultGradient(sizeCageX, coordinateForNodes, listForTraceOfMatrix, formCages, method='LQ')
        cloudGradX.append(cloudForGraduent[0])
        cloudGradY.append(cloudForGraduent[1])
        gradusBetta -= 1
    x = np.linspace(90, 0, 89)
    formPicture = searchMaxValuesForGrafics(formPicture, coordinateForNodes)
    drawXYGraduents(x, cloudGradX, 'blue', "Градиент Х", "X.png")
    drawXYGraduents(x, cloudGradY, 'green', "Градиент Y", "Y.png")
    drawGrid(coordinateForNodes, cloudAllNodes, sizeCageX, formPicture, formCages)
    return [cloudGradX, cloudGradY]

def compareResultsOfMethodsWithExactValue(exactValueForXY, resultOfMethod):
    cloudComparingX, cloudComparingY = [], []
    for i in range(len(resultOfMethod[0])):
        cloudComparingX.append(abs(resultOfMethod[0][i] - exactValueForXY[0]) / abs(exactValueForXY[0]))
        cloudComparingY.append(abs(resultOfMethod[1][i] - exactValueForXY[1]) / abs(exactValueForXY[1]))
    xy = np.linspace(90, 0, 89)
    drawXYGraduents(xy, cloudComparingX, 'blue', "Отношение численного и точного ответов", "X.png")
    drawXYGraduents(xy, cloudComparingY, 'red', "Отношение численного и точного ответов", "Y.png")

##########_Всё, что сязанно с отрисовкой_#########
def drawGrid(coordinateForNodes, cloudAllNodes, sizeCageX, formPicture, formCages):
    ### Отрисовка самой сетки
    fig, ax = plt.subplots()
    sizeOneCage = 0
    if formCages == 'rectangle': sizeOneCage = 5
    xPlot = np.zeros((sizeCageX * sizeCageX, sizeOneCage))
    yPlot = np.zeros((sizeCageX * sizeCageX, sizeOneCage))


    for i in range(len(cloudAllNodes)):
        for j in range(len(cloudAllNodes[0])):
            saver = int(cloudAllNodes[i][j])
            xPlot[i][j] = coordinateForNodes[saver][0]
            yPlot[i][j] = coordinateForNodes[saver][1]
            if j == 0:
                xPlot[i][sizeOneCage - 1] = coordinateForNodes[saver][0]
                yPlot[i][sizeOneCage -1] = coordinateForNodes[saver][1]
    for i in range(len(cloudAllNodes)):
        ax.plot(xPlot[i], yPlot[i], color='blue')
    plt.grid(True)
    plt.title("Сетка градиента")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-formPicture, formPicture)
    plt.ylim(-formPicture, formPicture)
    plt.savefig("fig.png")
    plt.close()

def drawXYGraduents(xy, cloudGrad, color, textY, saveFig):
    print(f"{cloudGrad}")
    if max(cloudGrad) > -0.01 and max(cloudGrad) < 0.01:
        maxValue = 0.025
        minValue = -0.025
    elif max(cloudGrad) < 0:
        maxValue = min(cloudGrad) * 1.25
        minValue = max(cloudGrad) * 0.75
    else:
        maxValue = max(cloudGrad) * 1.25
        minValue = min(cloudGrad) * 0.75
    plt.plot(xy, cloudGrad, color=color)
    plt.grid(True)
    # plt.axis('equal')
    plt.xlim(90, 0)

    if max(cloudGrad) > -0.01 and max(cloudGrad) < 0.01:  plt.ylim(minValue, maxValue)
    elif max(cloudGrad) < 0:      plt.ylim(maxValue, minValue)
    else:                         plt.ylim(minValue, maxValue)
    plt.title("Изменение градиента")
    plt.xlabel("Градус")
    plt.ylabel(textY)
    plt.savefig(saveFig)
    plt.close()

def searchMaxValuesForGrafics(massive, coordinateForNodes):
    ### Поиск максимального значения для выбора масштаба отрисовки графика в зависимости от изменения угла
    for i in coordinateForNodes:
        for j in i:
            if massive < abs(j):
                massive = abs(j) * 1.25
    return massive
#################################################

def startFunction(xy):
    return  3. * pow(xy[0] - 1, 3) + 2. * pow(xy[1] - 1, 4)

def getTriangleGreen(xyNow, xyBefore):

    ### Вычисление локальных градиентов в ячейках
    return [(1. / 2.) * (startFunction(xyNow) + startFunction(xyBefore)) * (xyNow[1] - xyBefore[1]),
            (-1. / 2.) * (startFunction(xyNow) + startFunction(xyBefore)) * (xyNow[0] - xyBefore[0])]

def square(sizeCageX, coordinateForGradients):
    return (abs(coordinateForGradients[0][0][0]) + abs(coordinateForGradients[0][sizeCageX - 1][0])) * (abs(coordinateForGradients[0][0][1]) + abs(coordinateForGradients[sizeCageX - 1][sizeCageX - 1][1]))

def runningForCageGreen(sizeCageX, coordinateForGradients):
    ### Здесь идём по сетке, рассчитывая значение локальных градиентов, попутно вычисляя их сумму
    cloudForLocalGraduents, buffer = [0., 0.], [0., 0.]
    for i in range(2):
        if i == 0:
            for j in range(1, sizeCageX):
                buffer = getTriangleGreen(coordinateForGradients[j][0], coordinateForGradients[j - 1][0])
                for z in range(len(buffer)):
                    cloudForLocalGraduents[z] += buffer[z]
            for j in range(1, sizeCageX):
                buffer = getTriangleGreen(coordinateForGradients[sizeCageX - 1][j], coordinateForGradients[sizeCageX - 1][j - 1])
                for z in range(len(buffer)):
                    cloudForLocalGraduents[z] += buffer[z]

        if i == 1:
            for j in range(sizeCageX - 2, -1, -1):
                buffer = getTriangleGreen(coordinateForGradients[j][2], coordinateForGradients[j + 1][2])
                for z in range(len(buffer)):
                    cloudForLocalGraduents[z] += buffer[z]
            for j in range(sizeCageX - 2, -1, -1):
                buffer = getTriangleGreen(coordinateForGradients[0][j], coordinateForGradients[0][j + 1])
                for z in range(len(buffer)):
                    cloudForLocalGraduents[z] += buffer[z]

    ### Вычмсляем значение градиента для XY компонент в центральной точке
    for i in range(len(cloudForLocalGraduents)):
        cloudForLocalGraduents[i] *= (1. / square(sizeCageX, coordinateForGradients))
    return cloudForLocalGraduents

def Lx(xyNow, xyCenter):
    return xyNow[0] - xyCenter[0]

def Ly(xyNow, xyCenter):
    return xyNow[1] - xyCenter[1]

def F(xyNow, xyCenter):
    return startFunction(xyNow) - startFunction(xyCenter)

def runningForCageLQ(sizeCageX, coordNow, centerOfCage):
    cloudForLocalGraduents, aXY = [], [0., 0.]
    lxx, lyy, lxy, fLx, fLy = 0., 0., 0., 0., 0.
    wk = 1.

    ### Здесь идём по сетке, рассчитывая значение компонент
    ### метода МНК для ах ау, попутно вычисляя их сумму
    for i in range(2):
        if i == 0:
            for j in range(1, sizeCageX, 1):
                lxx += wk * pow(Lx(coordNow[j][0], centerOfCage), 2)
                lyy += wk * pow(Ly(coordNow[j][0], centerOfCage), 2)
                lxy += wk * Lx(coordNow[j][0], centerOfCage) * Ly(coordNow[j][0], centerOfCage)
                fLx += wk * Lx(coordNow[j][0], centerOfCage) * F(coordNow[j][0], centerOfCage)
                fLy += wk * Ly(coordNow[j][0], centerOfCage) * F(coordNow[j][0], centerOfCage)

            for j in range(1, sizeCageX):
                lxx += wk * pow(Lx(coordNow[sizeCageX - 1][j], centerOfCage), 2)
                lyy += wk * pow(Ly(coordNow[sizeCageX - 1][j], centerOfCage), 2)
                lxy += wk * Lx(coordNow[sizeCageX - 1][j], centerOfCage) * Ly(coordNow[sizeCageX - 1][j], centerOfCage)
                fLx += wk * Lx(coordNow[sizeCageX - 1][j], centerOfCage) * F(coordNow[sizeCageX - 1][j], centerOfCage)
                fLy += wk * Ly(coordNow[sizeCageX - 1][j], centerOfCage) * F(coordNow[sizeCageX - 1][j], centerOfCage)

        if i == 1:
            for j in range(sizeCageX - 2, -1, -1):
                lxx += wk * pow(Lx(coordNow[j][2], centerOfCage), 2)
                lyy += wk * pow(Ly(coordNow[j][2], centerOfCage), 2)
                lxy += wk * Lx(coordNow[j][2], centerOfCage) * Ly(coordNow[j][2], centerOfCage)
                fLx += wk * Lx(coordNow[j][2], centerOfCage) * F(coordNow[j][2], centerOfCage)
                fLy += wk * Ly(coordNow[j][2], centerOfCage) * F(coordNow[j][2], centerOfCage)

            for j in range(sizeCageX - 2, -1, -1):
                lxx += wk * pow(Lx(coordNow[0][j], centerOfCage), 2)
                lyy += wk * pow(Ly(coordNow[0][j], centerOfCage), 2)
                lxy += wk * Lx(coordNow[0][j], centerOfCage) * Ly(coordNow[0][j], centerOfCage)
                fLx += wk * Lx(coordNow[0][j], centerOfCage) * F(coordNow[0][j], centerOfCage)
                fLy += wk * Ly(coordNow[0][j], centerOfCage) * F(coordNow[0][j], centerOfCage)

    ### Вычмсляем значение градиента для XY компонент в центральной точке
    aXY[0] = (lyy * (fLx) - lxy * (fLy)) / (lxx * lyy - pow(lxy, 2))
    aXY[1] = (lxx * (fLy) - lxy * (fLx)) / (lxx * lyy - pow(lxy, 2))
    return aXY

def getResultGradient(sizeCageX, coordinateForNodes , listForTraceOfMatrix, formCages, method):
    ### Здесь рассчитываем значение градиента для обоих методов
    quantity = 2
    coordinateForGradients = [[[0. for z in range(quantity)] for j in range(sizeCageX)] for i in range(sizeCageX)]

    ### Поиск центральных точек в ячейках
    for i in range(len(listForTraceOfMatrix)):
        for j in range(len(listForTraceOfMatrix[0])):
            cloudForListNodes = listForTraceOfMatrix[i][j]
            coordinateForGradients[i][j][0] = (1. / 4.) * (((coordinateForNodes[int(cloudForListNodes[0])][0]) + (coordinateForNodes[int(cloudForListNodes[1])][0])) + \
                                              ((coordinateForNodes[int(cloudForListNodes[2])][0]) + (coordinateForNodes[int(cloudForListNodes[3])][0])))
            coordinateForGradients[i][j][1] = (coordinateForNodes[int(cloudForListNodes[1])][1] + coordinateForNodes[int(cloudForListNodes[2])][1]) / 2.

    if method == 'green':   cloudForGraduents = runningForCageGreen(sizeCageX, coordinateForGradients)
    else:                   cloudForGraduents = runningForCageLQ(sizeCageX, coordinateForGradients, coordinateForNodes[0])
    return cloudForGraduents

def greenFunc(sizeCageX, hx, hy, gradusBetta, coordinateForNodes, cloudAllNodes, listForTraceOfMatrix, formCages):
    formPicture = 0.

    ### Расчёт градиента для определённого угла, а также отрисовка сетки,
    ### лучше комментировать, если хотите выполнить функцию countXYForGraficsXY_PNG
    cloudForGraduent = getResultGradient(sizeCageX, coordinateForNodes, listForTraceOfMatrix, formCages, method='green')
    formPicture = searchMaxValuesForGrafics(formPicture, coordinateForNodes)
    drawGrid(coordinateForNodes, cloudAllNodes, sizeCageX, formPicture, formCages)
    print(f'\nMethod Gauss-Green:\n cloudForGraduentX:\t{cloudForGraduent[0]}\n cloudForGraduentY\t{cloudForGraduent[1]}')

    ### Расчёт градиента для угла от 90 до 1 градуса. Отрисовка и сетки, и XY компонент в соответсвии с изменением угла
    # cloudForGraduent = countXYForGraficsXY_PNG(sizeCageX, hx, hy, gradusBetta, coordinateForNodes, cloudAllNodes, formCages)
    return cloudForGraduent

def leastSquares(sizeCageX, hx, hy, gradusBetta, coordinateForNodes, cloudAllNodes, listForTraceOfMatrix, formCages):
    formPicture = 0.

    ### Расчёт градиента для определённого угла, а также отрисовка сетки,
    ### лучше комментировать, если хотите выполнить функцию countXYForGraficsXY_PNG
    cloudForGraduent = getResultGradient(sizeCageX, coordinateForNodes, listForTraceOfMatrix, formCages, method='LQ')
    formPicture = searchMaxValuesForGrafics(formPicture, coordinateForNodes)
    drawGrid(coordinateForNodes, cloudAllNodes, sizeCageX, formPicture, formCages)
    print(f'\nMethod LQ:\n cloudForGraduentX:\t{cloudForGraduent[0]}\n cloudForGraduentY\t{cloudForGraduent[1]}')

    # ### Расчёт градиента для угла от 90 до 1 градуса. Отрисовка и сетки, и XY компонент в соответсвии с изменением угла
    # cloudForGraduent = countXYForGraficsXY_PNG(sizeCageX, hx, hy, gradusBetta, coordinateForNodes, cloudAllNodes, formCages)
    return cloudForGraduent

def Main():
    formCages = 'rectangle'
    sizeCageX = 3
    hx = hy = 0.01
    gradusBetta, formPicture, exactValueForXY = 90., [], [9., -8.]

    coordinateForNodes = getValuesForNodes(hx, hy, gradusBetta, sizeCageX, formCages)
    cloudAllNodes = nodesForGrid(sizeCageX, formCages)
    listForTraceOfMatrix = getTraceOfMatrix(sizeCageX, cloudAllNodes, formCages)

    resultGreen = greenFunc(sizeCageX, hx, hy, gradusBetta, coordinateForNodes, cloudAllNodes, listForTraceOfMatrix, formCages)
    resultLQ = leastSquares(sizeCageX, hx, hy, gradusBetta, coordinateForNodes, cloudAllNodes, listForTraceOfMatrix, formCages)

    ### Разкоментироват лишь в случае использования функции "countXYForGraficsXY_PNG" в
    ### leastSquares и greenFunc
    # compareResultsOfMethodsWithExactValue(exactValueForXY, resultLQ)


if __name__ == '__main__':
    Main()
