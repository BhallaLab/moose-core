import moose

if moose._moose.__generated_by__ != "pybind11":
    quit(0) 

class MyCompt(moose.Compartment):

    def __init__(self, path):
        super().__init__(path)
        self.temperature = 25.0
        assert self.temperature == 25.0, self.temperature
        assert self.path == super().path, (self.path, super().path)

    def __setattr__(self, k, v):
        try:
            super().setField(k, v)
        except Exception:
            self.__dict__[k] = v

    def __getattr__(self, k):
        try:
            return super().__getattr__(k)
        except Exception:
            return self.__dict__[k]
            

def test_inheritance():
    moose.Neutral('compt')
    c = MyCompt('/compt/abc')
    print(c)

def main():
    #  print(moose.Compartment.mro())
    #  print(MyCompt.mro())
    test_inheritance()

if __name__ == '__main__':
    main()
