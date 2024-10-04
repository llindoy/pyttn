
class system_bath_info:
    #the system info object here expects a single 
    def __init__(self, sysinf, Hs = None, baths =[]):
        self._sys = sysinf
        self._Hs = Hs
        self._baths = baths

    def system_hamiltonian(self, Hs)
        self._Hs = Hs

    def add_bath(self, bath):
        #function for attaching a bath to this object
        self._baths.append(bath)


    


