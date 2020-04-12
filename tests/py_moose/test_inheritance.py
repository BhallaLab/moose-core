import moose

class MyCompt(moose.Compartment):

    def __init__(self, path):
        moose.Compartment.__init__(self, path)
        print(self.__class__)
        self.temperature = 25.0
        assert self.temperature == 25.0

def test_inheritance():
    c = MyCompt('x')
    print(c)

def main():
    print(moose.Compartment.mro())
    print(MyCompt.mro())
    test_inheritance()

if __name__ == '__main__':
    main()
