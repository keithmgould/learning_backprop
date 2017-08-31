import math

LEARNING_RATE = 0.5

class NeuralNetwork:
  def __init__(self, inputNeurons, outputNeurons, hiddenNeurons):
    self.inputNeurons = inputNeurons
    self.outputNeurons = outputNeurons
    self.hiddenNeurons = hiddenNeurons

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
      print(outputNeuron.name, ": ", outputNeuron.output)

  def feedForward(self, forwardValues):
    for i, forwardValue in enumerate(forwardValues):
      self.inputNeurons[i].receiveSignal(forwardValue)

  def learn(self):
    for hiddenNeuron in self.hiddenNeurons:
      hiddenNeuron.updateRearWeights()
    for outputNeuron in self.outputNeurons:
      outputNeuron.updateRearWeights()

class Neuron:
  def __init__(self, bias, name):
    self.bias = bias
    self.name = name
    self.forwardAttachments = []
    self.rearAttachments = []
    self.total = 0
    self.output = 0
    self.rearSignalCount = 0

  def connectToForwardNeuron(self, forwardNeuron, weight):
    newAttachment = Attachment(self, forwardNeuron, weight)
    forwardNeuron.rearAttachments.append(newAttachment)
    self.forwardAttachments.append(newAttachment)

  def clearTotal(self):
    self.rearSignalCount = 0
    self.total = 0
    self.output = 0
    for forwardAttachment in self.forwardAttachments:
      forwardAttachment.forwardNeuron.clearTotal()

  def receiveSignal(self, newInput):
    self.rearSignalCount += 1
    self.total += newInput
    if self.allSignalsReceived():
      self.calculateOutput()
      self.fireForward()

  def allSignalsReceived(self):
    return self.rearSignalCount >= len(self.rearAttachments)

  def fireForward(self):
    for forwardAttachment in self.forwardAttachments:
      weightedOutput = self.output * forwardAttachment.weight
      forwardAttachment.forwardNeuron.receiveSignal(weightedOutput)

  # sigmoid function
  def squash(self, val):
    return 1 / (1 + math.exp(-val))

class InputNeuron(Neuron):
  def __init__(self, name):
    Neuron.__init__(self, 0, name)

  def calculateOutput(self):
    self.output = self.total

class HiddenNeuron(Neuron):
  def calculateOutput(self):
    self.output = self.squash(self.total + self.bias)

  # ------------------------------------------------
  # Methods below calculate partial derivatives

  # ∂E(total)/∂Weight(index)
  def calculate_pd_total_error_wrt_weight(self, index):
    a = self.calculate_pd_total_error_wrt_output()
    b = self.calculate_pd_output_wrt_net_input()
    c = self.calculate_pd_net_input_wrt_weight(index)
    return a * b * c

  # ∂E(total)/∂Out(self)
  def calculate_pd_total_error_wrt_output(self):
    pd_error = 0
    for forwardIndex, forwardAttachment in enumerate(self.forwardAttachments):
      pd_error += self.calculate_pd_error_wrt_output(forwardIndex)
    return pd_error

  # ∂E(forwardIndex)/∂Out(self) = [∂E(self)/∂Net(self)] * [∂Net(self)/∂Out(forwardIndex)]
  def calculate_pd_error_wrt_output(self, forwardIndex):
    forwardNeuron = self.forwardAttachments[forwardIndex].forwardNeuron
    a = forwardNeuron.calculate_pd_error_wrt_net_input()
    b = self.calculate_pd_net_input_wrt_output(forwardIndex)
    return a * b

  # ∂Out(self)/∂Net(self)
  def calculate_pd_output_wrt_net_input(self):
    return self.output * (1 - self.output)

  # ∂Net(self)/∂Out(self)
  def calculate_pd_net_input_wrt_output(self, index):
    return self.forwardAttachments[index].weight

  # ∂Net(self)/∂Weight(index)
  def calculate_pd_net_input_wrt_weight(self, index):
    return self.rearAttachments[index].rearNeuron.output

  def updateRearWeight(self, index):
    total_err_wrt_weight = self.calculate_pd_total_error_wrt_weight(index)
    oldWeight = self.rearAttachments[index].weight
    self.rearAttachments[index].weight -= LEARNING_RATE * total_err_wrt_weight
    print(self.name, 'is updating rearWeight(', index, ') from ', oldWeight, ' to ', self.rearAttachments[index].weight)

  def updateRearWeights(self):
    for rearIndex, rearAttachment in enumerate(self.rearAttachments):
      self.updateRearWeight(rearIndex)

class OutputNeuron(HiddenNeuron):
  def __init__(self, bias, name, target):
    HiddenNeuron.__init__(self, bias, name)
    self.target = target

  def calculateError(self):
    return 0.5 * (self.target - self.output) ** 2

  # ∂E(self)/∂Net(self)
  def calculate_pd_error_wrt_net_input(self):
    a = self.calculate_pd_total_error_wrt_output()
    b = self.calculate_pd_output_wrt_net_input()
    return a * b

  # ∂Error(self)/∂Out(self)
  def calculate_pd_total_error_wrt_output(self):
    return -(self.target - self.output)

class Attachment:
  def __init__(self, rearNeuron, forwardNeuron, weight):
    self.rearNeuron = rearNeuron
    self.forwardNeuron = forwardNeuron
    self.weight = weight

#-----------------------------------------------------
# Now lets actually use the network

# biases
b1 = 0.35
b2 = 0.60

# inputs
i1 = InputNeuron('i1')
i2 = InputNeuron('i2')

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

nn = NeuralNetwork([i1,i2], [o1, o2], [h1, h2])
nn.clearTotal()
nn.printOutput()
nn.feedForward([0.05, 0.10])
nn.printOutput()
print("Total Error: ", nn.calculateTotalError())
print("o1 Total Error wrt W5: ", o1.calculate_pd_total_error_wrt_weight(0));
print("o1 Total Error wrt W6: ", o1.calculate_pd_total_error_wrt_weight(1));
print("h1 Total Error wrt W1: ", h1.calculate_pd_total_error_wrt_weight(0));
nn.learn()
nn.clearTotal()
nn.feedForward([0.05, 0.10])
print("Total Error: ", nn.calculateTotalError())




