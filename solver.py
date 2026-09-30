class qm_system:
    def __init__(self, V, psi, dx):
        self.V = V
        self.psi = psi
        self.dx = dx
        self.x = 0

    def getNext(self):
        dx = self.dx
        x_0 = self.x


        # return (self.pos, self.vel, self.time)


def main():
    ...