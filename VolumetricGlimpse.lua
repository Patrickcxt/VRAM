local VolumetricGlimpse, parent = torch.class("nn.VolumetricGlimpse", "nn.Module")

function VolumetricGlimpse:__init(size, time, depth, scale)
    require 'nnx'
    if torch.type(size) == 'table' then
        self.height = size[1]
	self.width = size[2]
    else
        self.height, self.width = size, size
    end
    self.time = time or 16
    self.depth = depth or 3
    self.scale = scale or 2

    assert(torch.type(self.width) == 'number')
    assert(torch.type(self.height) == 'number')
    assert(torch.type(self.time) == 'number')
    assert(torch.type(self.depth) == 'number')
    assert(torch.type(self.scale) == 'number')
    parent.__init(self)
    self.gradInput = {torch.Tensor(), torch.Tensor()}
    if self.scale == 2 then
        self.module = nn.VolumetricAveragePooling(1, 2, 2, 1, 2, 2)
    else
        self.module = nn.VolumetricAveragePooling(1, 2, 2, 1, 2, 2)
    end
    self.modules = {self.module}
end

function VolumetricGlimpse:updateOutput(inputTable)
    assert(torch.type(inputTable) == 'table')
    assert(#inputTable >= 2)
    -- input = [batchsize x channels x time x height x width]
    -- location = [batchsize x 3 (l, x, y)]
    local inputVideos, location = unpack(inputTable)
    print(location)

    -- Get video clips
    local input = torch.Tensor(inputVideos:size(1), 3, self.time, inputVideos:size(4), inputVideos:size(5))
    for sampleIdx = 1, inputVideos:size(1) do
        local video = inputVideos[sampleIdx]
        local l = location[sampleIdx]:select(1, 1)
	l = (l+1) / 2  -- 0 ~ 1
	local stFrame, edFrame = self:getBoundFrame(video, l)
	input[sampleIdx] = video[{{}, {stFrame, edFrame}, {}, {}}]
    end
    
    self.output:resize(input:size(1), self.depth, input:size(2), self.time, self.height, self.width)
    self._crop = self._crop or self.output.new()
    self._pad = self._pad or input.new()

    for sampleIdx = 1, self.output:size(1) do
        local outputSample = self.output[sampleIdx]        
	local inputSample = input[sampleIdx]
	local lyx = location[sampleIdx]
	-- (-1, -1) top left corner, (1,1) bottom right conner of image
        local l, y, x = lyx:select(1, 1), lyx:select(1, 2), lyx:select(1, 3)
	-- (0, 0), (1, 1)
	y, x = (y+1) / 2, (x+1) / 2
	-- for each depth of glimpse: pad, crop, downscale
	local glimpseWidth = self.width
	local glimpseHeight = self.height
	for depth = 1, self.depth do
	    local dst = outputSample[depth]
	    if depth > 1 then
	        glimpseWidth = glimpseWidth * self.scale
		glimpseHeight = glimpseWidth * self.scale
	    end

	    -- add zero padding (glimpse could be partially out of bounds)
	    local padWidth = math.floor((glimpseWidth-1)/2)
	    local padHeight = math.floor((glimpseHeight-1)/2)
	    -- pad = [channel * time * height * width]
	    self._pad:resize(input:size(2), input:size(3), input:size(4)+padHeight*2, input:size(5)+padWidth*2):zero()
	    local center = self._pad:narrow(3, padHeight+1, input:size(4)):narrow(4, padWidth+1, input:size(5))
	    center:copy(inputSample)

            --crop it
	    local h, w = self._pad:size(3)-glimpseHeight, self._pad:size(4)-glimpseWidth
	    local y, x = math.min(h, math.max(0, y*h)), math.min(w, math.max(0, x*w))
	    if depth == 1 then
	        dst:copy(self._pad:narrow(3, y+1, glimpseHeight):narrow(4, x+1, glimpseWidth))
	    else
	        self._crop:resize(input:size(2), input:size(3), glimpseHeight, glimpseWidth)
		self._crop:copy(self._pad:narrow(3, y+1, glimpseHeight):narrow(4, x+1,glimpseWidth))
		if torch.type(self.module) == 'nn.VolumetricAveragePooling' then
		    local poolWidth = glimpseWidth / self.width
		    assert(poolWidth % 2 == 0)
		    local poolHeight = glimpseHeight / self.height
		    assert(poolHeight % 2 == 0)
		    self.module.kW = poolWidth
		    self.module.kH = poolHeight
		    self.module.dW = poolWidth
		    self.module.dH = poolHeight
		    self.module.kT = 1
		    self.module.dT = 1
		end
		dst:copy(self.module:updateOutput(self._crop))
	    end
	end
    end
    self.output:resize(input:size(1), self.depth*input:size(2), self.time, self.height, self.width)
    --self.output = self:fromBatch(self.output, 1)
    print("VolumetricGlimpse Completed!")
    return self.output
end

function VolumetricGlimpse:updateGradInput(inputTable, gradOutput)
    local inputVideos, location = unpack(inputTable)
    local gradInput, gradLocation = unpack(self.gradInput)
    --local input, location = self:toBatch(input, 3), self:toBatch(location, 1)  -- ???
    --local gradOutput = self:toBatch(gradOutput, 3)  -- ???

    gradInput:resizeAs(inputVideos):zero()
    gradLocation:resizeAs(location):zero()  -- no backprop through location

    gradOutput = gradOutput:view(inputVideos:size(1), self.depth, inputVideos:size(2), self.time, self.height, self.width)

    -- Get video clips

    for sampleIdx = 1, gradOutput:size(1) do
        local gradOutputSample = gradOutput[sampleIdx]
	local gradInputSample = gradInput[sampleIdx]
	local lyx = location[sampleIdx] -- frame, height, width
	local l, y, x = lyx:select(1, 1), lyx:select(1, 2), lyx:select(1, 3)
	l, y, x = (l+1)/2, (y+1)/2, (x+1)/2
	local stFrame, edFrame = self:getBoundFrame(inputVideos[sampleIdx], l)

	-- for each depth of glimpse: pad, crop , downscale
	local glimpseWidth = self.width
	local glimpseHeight = self.height
	for depth = 1, self.depth do
	    local src = gradOutputSample[depth]
	    if depth > 1 then
	        glimpseWidth = glimpseWidth * self.scale
		glimpseHeight = glimpseHeight * self.scale
	    end

	    -- add zero padding (glimpse could be partially out of bounds)
	    local padWidth = math.floor((glimpseWidth-1)/2)
	    local padHeight = math.floor((glimpseHeight-1)/2)
	    self._pad:resize(inputVideos:size(2), self.time, input:size(4)+padHeight*2, input:size(5)+padWidth*2):zero()

	    local h, w = self._pad:size(3) - glimpseHeight, self._pad:size(4) - glimpseWidth
	    local y, x = math.min(h, math.max(0, y*h)), math.min(w, math.max(0, x*w))
	    local pad = self._pad:narrow(3, y+1, glimpseHeight):narrow(4, x+1, glimpseWidth)

            -- upscale glimpse for different depths
	    if depth == 1 then
	        pad:copy(src)
	    else
	        self._crop:resize(inputVideos:size(2), self.time, glimpseHeight, glimpseWidth)
		if torch.type(self.module) == 'nn.VolumetricAveragePooling' then
		    local poolWidth = glimpseWidth / self.width
		    assert(poolWidth % 2 == 0)
		    local poolHeight = glimpseHeight / self.height
		    assert(poolHeight % 2 == 0)
		    self.module.kW = poolWidth
		    self.module.kH = poolHeight
		    self.module.dW = poolWidth
		    self.module.dH = poolHeight
		    self.module.kT = 1
		    self.module.dT = 1
		end
		pad:copy(self.module:updateGradInput(self._crop, src))
	    end

	    -- copy into gradInput tensor (excluding padding)
	    gradInputSample[{{}, {stFrame, edFrame}, {}, {}}]:add(self._pad:narrow(3, padHeight+1, input:size(4)):narrow(4, padWidth+1, input:size(5)))
	end
    end

    self.gradInput[1] = gradInput
    self.gradInput[2] = gradLocation

    return self.gradInput
    
end


function VolumetricGlimpse:getBoundFrame(video, l)
    local frameIdx = math.floor(l * video:size(2))
    local stFrame, edFrame = frameIdx-self.time/2+1, frameIdx+self.time/2
    if stFrame < 1 then
        stFrame, edFrame = 1, self.time
    end 
    if edFrame > video:size(2) then
	stFrame, edFrame = video:size(2)-self.time+1, video:size(2)
    end
	--input[sampleIdx] = video[{{}, {stFrame, edFrame}, {}, {}}]
    return stFrame, edFrame
end



