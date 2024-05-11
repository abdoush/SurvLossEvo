import numpy as np
import pandas as pd
import deap
from deap import tools
from deap import base, creator
import time
from random import randrange
import copy
import random
from Genetic import Representation
# import MNIST
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


class NeuroEvolution:

    @staticmethod
    def treeCrossover(ind1, ind2):
        kmap = {2: [1, 3, 5, 6], 3: [2, 4, 7, 8], 4: [3, 5, 6], 5: [4, 7, 8]}
        crossPoint = random.choice([2, 3, 4, 5])
        changedIndexes = kmap[crossPoint]

        oldInd1 = copy.copy(ind1)

        for idx in changedIndexes:
            ind1[idx] = ind2[idx]
            ind2[idx] = oldInd1[idx]

        return ind1, ind2

    @staticmethod
    def HUX(ind1, ind2, fixed=True):
        # index variable
        idx = 0

        # Result list
        res = []

        # With iteration
        for i in ind1:
            if i != ind2[idx]:
                res.append(idx)
            idx = idx + 1
        if (len(res) > 1):
            numberOfSwapped = randrange(1, len(res))
            if (fixed):
                numberOfSwapped = len(res) // 2
            indx = random.sample(res, numberOfSwapped)

            oldInd1 = copy.copy(ind1)

            for i in indx:
                ind1[i] = ind2[i]

            numberOfSwapped = randrange(1, len(res))
            if (fixed):
                numberOfSwapped = len(res) // 2
            indx = random.sample(res, numberOfSwapped)

            for i in indx:
                ind2[i] = oldInd1[i]
        return ind1, ind2

    @staticmethod
    def evaluate(individual, model, epochs=10, callbaks=None, optimizationTask='step', evaluation='validation'):
        #print('evaluation')
        if (optimizationTask == 'step'):
            model.f_step = Representation.Function(individual)
        elif(optimizationTask == 'loss'):
            model.f_loss = Representation.Function(individual)
        model.create_model()
        batchSize = 128
        loss = "categorical_crossentropy"
        optimizer = "adam"
        metric = "accuracy"
        patience = 10
        verbose = 0
        # model.setConfig(loss, optimizer, metric, batchSize, epochs, patience, verbose)
        model.set_config(optimizer, batch_size=batchSize, epochs=epochs, test_mode=True, verbose=verbose, callbacks=callbaks)
        model.fit()
        np.random.seed()
        random.seed()
        if (evaluation == 'test'):
            # score = model.model.evaluate(model.X_test, model.y_test, verbose=0)
            score = model.evaluate_test()
            return score
        if (evaluation == 'train'):
            if (pd.isna(max(model.history.history['cindex']))):
                return 0.0,  # -1 * np.inf,
            else:
                return max(model.history.history['cindex']),
        #print(model.history.history.keys())
        #print(model.history.history['val_cindex'])
        #print(max(model.history.history['val_cindex']))
        if (pd.isna(max(model.history.history['val_cindex']))):
            return 0.0, #-1 * np.inf,
        else:
            return max(model.history.history['val_cindex']),

    @staticmethod
    def createPopulation(populationSize, indSize, include=False):
        pop = []
        np.random.seed()
        random.seed()
        relu = deap.creator.Individual([5, 7, 9, 1, 1, 7, 7, 9, 7])
        if (include):
            include = deap.creator.Individual(include)
        for i in range(populationSize):
            pop.append(deap.creator.Individual(
                [random.randint(1, 6), random.randint(7, 30), random.randint(7, 30), random.randint(1, 6),
                 random.randint(1, 6),
                 random.randint(7, 30), random.randint(7, 30), random.randint(7, 30), random.randint(7, 30)]))
        if (include):
            del pop[-1]
            pop.append(include)
        return list(pop)

    @staticmethod
    def mutateOperation(ind, numberOfFlipped=1, pruning=0.1):
        kmap = {2: [1, 3, 5, 6], 3: [2, 4, 7, 8], 4: [3, 5, 6], 5: [4, 7, 8]}
        crossPoint = random.choice([2, 3, 4, 5])
        changedIndexes = kmap[crossPoint]
        # print('before', ind)
        if random.random() < pruning:
            # print('pruning')
            for idx in changedIndexes:
                if (idx in [3, 4]):
                    ind[idx] = 1
                else:
                    ind[idx] = 7
        else:
            # print('mutate')
            for _ in range(numberOfFlipped):
                flipedOperation = random.randint(0, len(ind) - 1)
                if (flipedOperation in [0, 3, 4]):
                    ind[flipedOperation] = random.randint(1, 6)
                else:
                    ind[flipedOperation] = random.randint(7, 30)
        # print('after', ind)
        # print()
        return ind

    @staticmethod
    def createToolbox(indSize, model, alg='CHC', epochs=10, optimizationTask='step', evaluation='validation'):
        toolbox = base.Toolbox()
        if (alg == 'CHC'):
            toolbox.register("mate", NeuroEvolution.HUX)
        elif (alg == 'GA'):
            toolbox.register("mate", NeuroEvolution.treeCrossover)
        toolbox.register("mutate", NeuroEvolution.mutateOperation)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", NeuroEvolution.evaluate, model=model, epochs=epochs, optimizationTask=optimizationTask, evaluation=evaluation)
        return toolbox

    @staticmethod
    def CHC(model, population=False, populationSize=40, d=False, divergence=0.35, epochs=10,
            maxGenerations=np.inf, maxNochange=np.inf, timeout=np.inf, optimizationTask='step', evaluation='validation',
            stop=1, verbose=0, include=False):
        start = time.time()
        end = time.time()

        indSize = 9
        toolbox = NeuroEvolution.createToolbox(indSize, model, 'CHC', epochs, optimizationTask, evaluation)
        if (not population):
            population = NeuroEvolution.createPopulation(populationSize, indSize, include)
            # calculate fitness tuple for each individual in the population:
            fitnessValues = list(map(toolbox.evaluate, population))
            for individual, fitnessValue in zip(population, fitnessValues):
                individual.fitness.values = fitnessValue

        generationCounter = 0


        # extract fitness values from all individuals in population:
        fitnessValues = [individual.fitness.values[0] for individual in population]

        # initialize statistics accumulators:
        maxFitnessValues = []
        meanFitnessValues = []

        best = 0 # -1 * np.inf
        noChange = 0
        evaulationCounter = populationSize

        d0 = len(population[0]) // 2
        #if (not d):
        #    d = d0

        populationHistory = []
        for ind in population:
            populationHistory.append(ind)

        logDF = pd.DataFrame(
            columns=(
            'generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution', 'd'))

        # main evolutionary loop:
        # stop if max fitness value reached the known max value
        # OR if number of generations exceeded the preset value:
        while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
                end - start) < timeout:
            # update counter:
            generationCounter = generationCounter + 1

            for ind in population:
                print(ind, ind.fitness.values)
            print()

            # apply the selection operator, to select the next generation's individuals:
            offspring = toolbox.select(population, len(population))
            # clone the selected individuals:
            offspring = list(map(toolbox.clone, offspring))
            random.shuffle(offspring)

            newOffspring = []

            newOffspringCounter = 0

            # apply the crossover operator to pairs of offspring:
            numberOfPaired = 0
            numberOfMutation = 0
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if NeuroEvolution.hammingDistance(child1, child2) > d:
                    # print('Before')
                    # print(child1)
                    # print(child2)
                    toolbox.mate(child1, child2)
                    numberOfPaired += 1
                    newOffspringCounter += 2
                    addChild = True
                    for ind in populationHistory:
                        if (NeuroEvolution.hammingDistance(ind, child1) == 0):
                            newOffspringCounter -= 1
                            addChild = False
                            break
                    if (addChild):
                        populationHistory.append(child1)
                        newOffspring.append(child1)
                    addChild = True
                    for ind in populationHistory:
                        if (NeuroEvolution.hammingDistance(ind, child2) == 0):
                            newOffspringCounter -= 1
                            addChild = False
                            break
                    if (addChild):
                        populationHistory.append(child2)
                        newOffspring.append(child2)
                    # print('history length', len(populationHistory))
                    # print('After')
                    # print(child1)
                    # print(child2)
                    # print()
                    del child1.fitness.values
                    del child2.fitness.values
            #print('this is d', d)
            if (d == 0):
                d = d0
                newOffspring = []
                bestInd = tools.selBest(population, 1)[0]
                while (numberOfMutation < len(population)):
                    mutant = toolbox.clone(bestInd)
                    numberOfMutation += 1
                    toolbox.mutate(mutant, divergence)
                    newOffspring.append(mutant)
                    del mutant.fitness.values

            # if (newOffspringCounter == 0 and d > 0):
            #    d -= 1

            # calculate fitness for the individuals with no previous calculated fitness value:
            freshIndividuals = [ind for ind in newOffspring if not ind.fitness.valid]
            freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
            for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
                individual.fitness.values = fitnessValue

            evaulationCounter = evaulationCounter + len(freshIndividuals)

            if (numberOfMutation == 0):
                oldPopulation = copy.copy(population)
                population[:] = tools.selBest(population + newOffspring, populationSize)
                differentPopulation = False
                for index in range(0, len(population)):
                    if (NeuroEvolution.hammingDistance(oldPopulation[index], population[index]) != 0):
                        differentPopulation = True
                print(differentPopulation)
                if (not differentPopulation):
                    d -= 1
            else:
                bestInd = tools.selBest(population, 1)
                population[:] = tools.selBest(bestInd + newOffspring, populationSize)

            # collect fitnessValues into a list, update statistics and print:
            fitnessValues = [ind.fitness.values[0] for ind in population]

            maxFitness = max(fitnessValues)
            if (best >= maxFitness):
                noChange += 1
            if (best < maxFitness):
                best = maxFitness
                noChange = 0
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)

            end = time.time()

            # find and print best individual:
            best_index = fitnessValues.index(max(fitnessValues))
            if (verbose):
                print("Best Individual = ", np.round(maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
            # print()
            print(np.round(maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:',
                  numberOfMutation, ' d:', d)
            #print('new', newOffspringCounter)
            print()
            end = time.time()
            row = [generationCounter, (end - start), np.round(maxFitness, 2), meanFitness, evaulationCounter,
                   population[best_index], d]
            logDF.loc[len(logDF)] = row

        end = time.time()
        return logDF, population

    @staticmethod
    def GA(model, population=False, populationSize=40, crossOverP=0.9, mutationP=0.1, zeroP=0.5, maxGenerations=np.inf,
           maxNochange=np.inf,
           timeout=np.inf, stop=np.inf, verbose=0):

        start = time.time()
        end = time.time()

        indSize = 9
        toolbox = NeuroEvolution.createToolbox(indSize, model, 'GA')
        if (not population):
            population = NeuroEvolution.createPopulation(populationSize, indSize)

        generationCounter = 0
        # calculate fitness tuple for each individual in the population:
        fitnessValues = list(map(toolbox.evaluate, population))
        for individual, fitnessValue in zip(population, fitnessValues):
            individual.fitness.values = fitnessValue

        # extract fitness values from all individuals in population:
        fitnessValues = [individual.fitness.values[0] for individual in population]

        # initialize statistics accumulators:
        maxFitnessValues = []
        meanFitnessValues = []

        best = -1 * np.inf
        noChange = 0
        evaulationCounter = populationSize

        logDF = pd.DataFrame(
            columns=('generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations', 'best_solution'))

        # main evolutionary loop:
        # stop if max fitness value reached the known max value
        # OR if number of generations exceeded the preset value:
        while best < stop and generationCounter < maxGenerations and noChange < maxNochange and (
                end - start) < timeout:
            generationCounter = generationCounter + 1

            for ind in population:
                print(ind, ind.fitness.values)
            print()

            # apply the selection operator, to select the next generation's individuals:
            offspring = toolbox.select(population, len(population))
            # clone the selected individuals:
            offspring = list(map(toolbox.clone, offspring))

            # apply the crossover operator to pairs of offspring:

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossOverP:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutationP:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # calculate fitness for the individuals with no previous calculated fitness value:
            freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
            freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
            for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
                individual.fitness.values = fitnessValue

            evaulationCounter = evaulationCounter + len(freshIndividuals)

            population[:] = tools.selBest(population + offspring, populationSize)

            # collect fitnessValues into a list, update statistics and print:
            fitnessValues = [ind.fitness.values[0] for ind in population]

            maxFitness = max(fitnessValues)
            if (best >= maxFitness):
                noChange += 1
            if (best < maxFitness):
                best = maxFitness
                noChange = 0
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)

            end = time.time()

            # find and print best individual:
            best_index = fitnessValues.index(max(fitnessValues))
            if (verbose):
                print("Best Individual = ", np.round(maxFitness, 2), ", Gen = ", generationCounter, '\r', end='')
            # print()
            # print(np.round(100*maxFitness, 2), 'number of paired:', numberOfPaired, 'number of mutations:', numberOfMutation, ' d:', d)
            # print()
            end = time.time()
            row = [generationCounter, (end - start), np.round(100 * maxFitness, 2), meanFitness, evaulationCounter,
                   population[best_index]]
            logDF.loc[len(logDF)] = row

        end = time.time()
        return logDF, population

    @staticmethod
    def hammingDistance(ind1, ind2):
        ind1 = np.array(ind1)
        ind2 = np.array(ind2)
        return (len(ind1) - (np.sum(np.equal(ind1, ind2))))

    @staticmethod
    def SAGA(model, populationSize=40, reductionRate=0.5, pateince=2, step=10, d=False, divergence=3, epochs=10,
             targetFitness=1,
             verbose=0, qualOnly=False, timeout=np.inf, include=False,
             noChange=np.inf, evaluation='validation'):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness',
                                      'best_solution', 'surrogate_level', 'epochs'))
        partialModel = copy.copy(model)
        sampleSize = 100
        # partialModel.setTrainingSample(sampleSize)
        v_epochs = 2
        task = 'feature_selection'
        indSize = 9
        population = NeuroEvolution.createPopulation(populationSize, indSize, include)

        bestTrueFitnessValue = 0 # np.inf
        sagaActivationFunction = [1] * 9
        qual = False

        numberOfEvaluations = 0
        generationCounter = 0
        maxAllowedSize = int(partialModel.x_train.shape[0])

        d = indSize // 2
        surrogateLevel = 0

        pateince0 = pateince

        while (bestTrueFitnessValue < targetFitness and pateince > 0 and v_epochs <= epochs): #sampleSize < maxAllowedSize):
            # print('patience:', pateince)
            if (verbose):
                print('Current epochs:', v_epochs)
                # print('Current Approx Sample Size:', sampleSize)
                print('Current Population Size:', populationSize)
            pateince -= 1
            log, population = NeuroEvolution.CHC(model,
                                                 population,
                                                 d=d,
                                                 divergence=divergence,
                                                 epochs=v_epochs,
                                                 populationSize=populationSize,
                                                 maxNochange=step,
                                                 verbose=verbose)
            generationCounter = generationCounter + int(log.iloc[-1]['generation'])
            activationFunctionIndividual = log.iloc[-1]['best_solution']

            approxBestInGeneration = np.round(
                NeuroEvolution.evaluate(activationFunctionIndividual, model, v_epochs), 2)[0]
            end = time.time()

            # Check if the original value improved
            if (sagaActivationFunction != activationFunctionIndividual):
                pateince = pateince0
                bestTrueFitnessValue = approxBestInGeneration
                sagaActivationFunction = activationFunctionIndividual
                sagaIndividual = tools.selBest(population, 1)
                row = [generationCounter, (end - start), bestTrueFitnessValue,
                       sagaActivationFunction, surrogateLevel, v_epochs]
                logDF.loc[len(logDF)] = row
                if (verbose):
                    print('The best individual is saved', bestTrueFitnessValue)
                    print(row)

            v_epochs = v_epochs * 2
            populationSize = int(populationSize * reductionRate)
            surrogateLevel += 1
            d = indSize // 2
            # partialModel.setTrainingSample(sampleSize)
            newInd = NeuroEvolution.createPopulation(populationSize, indSize)

            population[:] = tools.selBest(sagaIndividual + newInd, populationSize)

        return logDF, population

    @staticmethod
    def SAGA_V0(model, populationSize=40, reductionRate=0.5, pateince=0, tolerance=0, maxNoChange=np.inf, d=False, divergence=3, epochs=10,
             targetFitness=1, initilizationMax=2, optimizationTask='step',
             verbose=0, qualOnly=False, timeout=np.inf, include=False,
             noChange=np.inf, evaluation='validation'):

        start = time.time()
        logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness',
                                      'best_solution', 'surrogate_level', 'epochs'))
        partialModel = copy.copy(model)
        sampleSize = 100
        # partialModel.setTrainingSample(sampleSize)
        v_epochs = 2
        task = 'feature_selection'
        indSize = 9
        population = NeuroEvolution.createPopulation(populationSize, indSize, include)
        for individual in population:
            individual.fitness.values = NeuroEvolution.evaluate(individual, model=model, epochs=v_epochs, optimizationTask=optimizationTask, evaluation=evaluation)

        bestTrueFitnessValue = 0 # np.inf
        sagaActivationFunction = [1] * 9
        currentFunction = [1] * 9
        qual = False
        #callbacks = [EarlyStopping(monitor='val_cindex', patience=100, restore_best_weights=True, mode='max')]

        if (include):
            bestTrueFitnessValue = np.round(
                NeuroEvolution.evaluate(include, model=model, epochs=epochs, optimizationTask=optimizationTask, evaluation=evaluation), 4)[0]
            sagaIndividual = deap.creator.Individual(include)
            sagaIndividual.fitness.values = bestTrueFitnessValue,
            print('Initial individual fitness is:', bestTrueFitnessValue)
            #print(sagaIndividual, bestTrueFitnessValue)
            #print(type(sagaIndividual), sagaIndividual, sagaIndividual.fitness.values)

        numberOfEvaluations = 0
        generationCounter = 0
        maxAllowedSize = int(partialModel.x_train.shape[0])

        d = indSize // 2
        surrogateLevel = 0

        pateinceCounter = pateince
        improvedInLevel = False
        maxNoChangeCounter = 0
        tolernaceCounter = tolerance

        initilizationCounter = 0

        if (verbose):
            print('Current epochs:', v_epochs)
            # print('Current Approx Sample Size:', sampleSize)
            print('Current Population Size:', populationSize)

        while (bestTrueFitnessValue < targetFitness and pateinceCounter >= 0 and v_epochs <= epochs): #sampleSize < maxAllowedSize):
            print('patience:', pateinceCounter, ' improved:', improvedInLevel, ' maxnochange:', maxNoChangeCounter, ' initilizationCounter:', initilizationCounter,  'tolerance:', tolernaceCounter)
            log, population = NeuroEvolution.CHC(model,
                                                 population,
                                                 d=d,
                                                 divergence=divergence,
                                                 epochs=v_epochs,
                                                 populationSize=populationSize,
                                                 maxGenerations=1,
                                                 optimizationTask=optimizationTask,
                                                 evaluation=evaluation,
                                                 verbose=0)
            generationCounter = generationCounter + int(log.iloc[-1]['generation'])
            activationFunctionIndividual = log.iloc[-1]['best_solution']
            d = log.iloc[-1]['d']
            #print('this is SAGA d', d)


            end = time.time()

            # Check if the original value improved
            if (currentFunction != activationFunctionIndividual):
                currentFunction = activationFunctionIndividual
                trueBestInGeneration = np.round(
                    NeuroEvolution.evaluate(activationFunctionIndividual, model=model, epochs=epochs, optimizationTask=optimizationTask, evaluation=evaluation), 4)[0]
                if (trueBestInGeneration > bestTrueFitnessValue):
                    improvedInLevel = True
                    maxNoChangeCounter = 0
                    initilizationCounter = 0
                    bestTrueFitnessValue = trueBestInGeneration
                    pateinceCounter = pateince
                    sagaActivationFunction = activationFunctionIndividual
                    sagaIndividual = tools.selBest(population, 1)[0]
                    row = [generationCounter, (end - start), bestTrueFitnessValue,
                           sagaActivationFunction, surrogateLevel, v_epochs]
                    logDF.loc[len(logDF)] = row
                    if (verbose):
                        print('The best individual is saved', bestTrueFitnessValue)
                        print(row)
                elif (trueBestInGeneration < bestTrueFitnessValue):
                    tolernaceCounter -= 1
                    if (verbose):
                        print('False Optimum:', trueBestInGeneration)
                        print(currentFunction)

            else:
                maxNoChangeCounter+=1

            if (d == 0):
                initilizationCounter += 1

            if (maxNoChangeCounter > maxNoChange or tolernaceCounter < 0 or initilizationCounter > initilizationMax):
                #print(type(sagaIndividual), sagaIndividual)
                if (not improvedInLevel):
                    pateinceCounter -= 1
                else:
                    pateinceCounter = pateince
                maxNoChangeCounter = 0
                initilizationCounter = 0
                tolernaceCounter = tolerance
                improvedInLevel = False
                currentFunction = sagaIndividual

                v_epochs = v_epochs * 2
                populationSize = int(populationSize * reductionRate)
                surrogateLevel += 1
                d = indSize // 2
                cycle = 1
                # partialModel.setTrainingSample(sampleSize)
                newInd = NeuroEvolution.createPopulation(populationSize, indSize)
                newInd.append(sagaIndividual)
                for individual in newInd:
                    individual.fitness.values = NeuroEvolution.evaluate(individual, model=model, epochs=v_epochs,
                                                                        optimizationTask=optimizationTask,
                                                                        evaluation=evaluation)
                #for ind in newInd:
                #    print(ind, ind.fitness.values)

                population[:] = tools.selBest(newInd, populationSize)

                if (verbose):
                    print('Current epochs:', v_epochs)
                    # print('Current Approx Sample Size:', sampleSize)
                    print('Current Population Size:', populationSize)

        return logDF, population

