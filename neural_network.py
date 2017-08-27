import math

class NeuralNetwork:
  def __init__(self, inputNeurons, outputNeurons):
    self.inputNeurons = inputNeurons
    self.outputNeurons = outputNeurons

  def clearTotal(self):
    for inputNeuron in self.inputNeurons:
      inputNeuron.clearTotal()

  def calculateTotalError(self):
    totalError = 0
    for outputNeuron in self.outputNeurons:
      totalError += outputNeuron.calculateError();
    return totalError

  def printOutput(self):
    print("Output:")
    for outputNeuron in self.outputNeurons:
      print(outputNeuron.final)

  def feedForward(self, forwardValues):
    for i, forwardValue in enumerate(forwardValues):
      self.inputNeurons[i].receiveSignal(forwardValue)

class Neuron:
  def __init__(self, bias, name):
    self.bias = bias
    self.name = name
    self.forwardAttachments = []
    self.rearAttachments = []
    self.total = 0
    self.final = 0
    self.rearSignalCount = 0

  def connectToForwardNeuron(self, neuron, weight):
    neuron.rearAttachments.append(Attachment(self, weight))
    self.forwardAttachments.append(Attachment(neuron, weight))

  def clearTotal(self):
    self.rearSignalCount = 0
    self.total = 0
    self.final = 0
    for forwardAttachment in self.forwardAttachments:
      forwardAttachment.neuron.clearTotal()

  def receiveSignal(self, newInput):
    self.rearSignalCount += 1
    self.total += newInput
    if self.allSignalsReceived():
      self.calculateFinal()
      self.fireForward()

  def allSignalsReceived(self):
    return self.rearSignalCount >= len(self.rearAttachments)

  def calculateFinal(self):
    final = self.total + self.bias
    if self.shouldSquash():
      final = self.squash(final)
    self.final = final

  def fireForward(self):
    for forwardAttachment in self.forwardAttachments:
      weighted = self.final * forwardAttachment.weight
      print(self.name, " firing to ", forwardAttachment.neuron.name, " with final: ", self.final, " and after weight: ", weighted)
      forwardAttachment.neuron.receiveSignal(weighted)

  def squash(self, val):
    return 1 / (1 + math.exp(-val))

class InputNeuron(Neuron):
  def shouldSquash(self):
    return False

class HiddenNeuron(Neuron):
  def shouldSquash(self):
    return True


class OutputNeuron(HiddenNeuron):
  def __init__(self, bias, name, target):
    HiddenNeuron.__init__(self, bias, name)
    self.target = target


  def calculateError(self):
    return 0.5 * (self.target - self.final) ** 2

class Attachment:
  def __init__(self, neuron, weight):
    self.neuron = neuron
    self.weight = weight

# biases
b1 = 0.35
b2 = 0.60

# inputs
i1 = InputNeuron(0, 'i1')
i2 = InputNeuron(0, 'i2')

# hiddens
h1 = HiddenNeuron(b1, 'h1')
h2 = HiddenNeuron(b1, 'h2')

# outputs
o1 = OutputNeuron(b2, 'o1', .01)
o2 = OutputNeuron(b2, 'o2', 0.99)

# connections
i1.connectToForwardNeuron(h1, .15)
i1.connectToForwardNeuron(h2, .25)
i2.connectToForwardNeuron(h1, .20)
i2.connectToForwardNeuron(h2, .30)
h1.connectToForwardNeuron(o1, .40)
h1.connectToForwardNeuron(o2, .50)
h2.connectToForwardNeuron(o1, .45)
h2.connectToForwardNeuron(o2, .55)

nn = NeuralNetwork([i1,i2], [o1, o2])
nn.clearTotal()
nn.printOutput()
nn.feedForward([0.05, 0.10])
nn.printOutput()
print(nn.calculateTotalError())

