# Define the Value

class Value:
    # Value contains a scalar value
    # with data, children, grad attributes
    # and methods for operations and their gradients

    def __init__(self, _data, _children=(), _op=''):
        self.data = _data
        self.grad = 0 # initally the gradient of a value is 0
        
        #internal variables for the autograd graph construction
        self._backward = lambda: None # The function that produced this node - initally set to None
        self._prev = set(_children) # the children that produced this node
        self._op = _op # the operation that produced this node
    

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Ensure other is a value object 
        out = Value(self.data + other.data, (self, other), '+') # Produces a new node formed from data and other (i.e. children)

        def _backward():
            self.grad += out.grad # by chain rule 1 * previous gradients as additions produces grad = 1... 
            other.grad += out.grad 
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # Ensure other is a value object 
        out = Value(self.data * other.data, (self, other), '*') # Produces a new node formed from data and other (i.e. children)

        def _backward():
            self.grad +=  other.grad * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward() 

        return out
    
    def __pow__(self, other):

        assert isinstance(other, (int, float), "only supporting int/float powers for now")
        out = Value(self.data**other.data, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data **(other - 1)) * out.grad
        out._backward = _backward 
       
        return out 
    
    def relu(self):
        if self.data < 0:
            out = Value(0, (self), "ReLU")
        else:
            out = self

        def _backward():
            self.grad += out.grad 
        out._backward = _backward()

        return out

    def backward(self):
        # perform backward pass on children elements 

        # topological order all of the children in the graph

        topo = [] 
        visited = set() 
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._pred:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Visit one variable at a time and apply the chain rule to calculate its gradient

        self.grad = 1 
        for v in reversed(topo):
            v._backward()

        def __neg__(self):
            return self*-1
        
        def __radd__(self, other): # add other and self Value objects
            return self + other

        def __sub__(self, other): # self - other
            return self + (-other)
        
        def __rsub__(self, other): # other - self 
            return other + (-self)
        
        def __rmul__(self, other): # other * self 
            return other * self 
        
        def __truediv__(self, other):
            return self * other**-1 
        
        def __rtruediv(self, other): 
            return other * self**-1 
        
        def __repr__(self): # representation string 
            return f"Value(data={self.data}, grad={self.grad})" 
    