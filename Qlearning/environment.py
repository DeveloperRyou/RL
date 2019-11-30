import itertools
import numpy as np

class gameOb():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

class gameEnv():
    def __init__(self,partial,size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        a = self.reset()
        plt.imshow(a, interpolation="nearest")

    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(), 1, 1, 2, None, 'hero')
        self.objects.append(hero)
        bug = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(bug)
        hole = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole)
        state = self.renderEnv()
        self.state = state
        return state

    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables)
            points.append(t)

        currentPosition = []
        for object in self.objects:
            if (object.x, object.y) not in currentPosition:
                currentPosition.append((object.x, object,y))

        for pos in currentPosition:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def moveChar