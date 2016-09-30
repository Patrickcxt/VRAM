require 'utils'
local DetLossCriterion, parent = torch.class("nn.DetLossCriterion", "nn.Criterion")

function DetLossCriterion:__init(criterion)
    parent.__init(self)
    self.criterion = criterion or nn.SmoothL1Criterion()
end

function DetLossCriterion:updateOutput(inputTable, target)
    self.gts = utils.get_gts(inputTable, target)
    print(self.gts)
    self.output = self.criterion:updateOutput(inputTable, self.gts)
    return self.output
end

function DetLossCriterion:updateGradInput(inputTable, target)
    self.gradInput = self.criterion:updateGradInput(inputTable, self.gts)
    return self.gradInput
end

