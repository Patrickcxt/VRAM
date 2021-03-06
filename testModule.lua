require 'dp'
require 'nn'
require 'VolumetricGlimpse'
require 'DetReward'
require 'DetLossCriterion'
require 'DataHandler'

-- test DataHandler
--[[
local dh = DataHandler()
trainset = dh:getTrainSet()
valset = dh:getValSet()
testset = dh:getTestSet()
print(trainset)
io.read()
print(valset)
io.read()
print(testset)
]]

-- test DetLossCriterion Modul
--[[
input = torch.Tensor({{2, 4}, {3, 8}})
target = {torch.Tensor({{1, 3}, {4, 5}}), torch.Tensor({{2, 10}})}
detreward = nn.DetLossCriterion()
output = detreward:forward(input, target)
print(output)
]]


-- test DetReward Module
--[[
input = {torch.Tensor({{2, 4}, {3, 8}}), torch.ones(3, 1)}
target = {torch.Tensor({{1, 3}, {4, 5}}), torch.Tensor({{2, 10}})}
detreward = nn.DetReward(nil, 1, nil)
output = detreward:forward(input, target)
print(output)
]]


-- test VolumetricGlimpse module
video = {}
for i = 1, 5 do
    table.insert(video, torch.Tensor(3, 32+i, 28, 28))
end
print(video)
location = torch.Tensor(20, 3)
input = {video, location}
model = nn.Sequential()
model:add(nn.VolumetricGlimpse({8, 8}, 16, 3, 2))
output = model:forward(input)
print(#output)
