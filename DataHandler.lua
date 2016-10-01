require 'utils'
local cjson = require 'cjson'
require 'image'

local DataHandler = torch.class("DataHandler")

function DataHandler:__init(root)
    self.root = root or "/home/t5/cxt/VRAM/data/ActivityNet/"
    self.indexPath = self.root .. "scheme/index.txt"
    self.gtPath = self.root .. "scheme/activity_net.v1-3.min.json"
    self.videoList, self.cutNameList = self:create_video_list()
    self.database, self.version, self.taxonomy = self:parse_json()
    self.root = self:create_taxonomy_tree()
    self.name2label, self.label2name = self:get_classes()
    self.trainSet, self.valSet, self.testSet = self:create_dataset()
end

function DataHandler:getTrainSet()
    return self.trainSet
end

function DataHandler:getValSet()
    return self.valSet
end

function DataHandler:getTestSet()
    return self.testSet
end

function DataHandler:create_video_list()
    local f = io.open(self.indexPath, 'r') 
    videoList, cutNameList = {}, {}
    assert(f)
    local cnt = 0
    for line in f:lines() do
        local re = utils.split(line, ",")
        table.insert(videoList, re[3])
	table.insert(cutNameList, self:cut_name(re[3]))
    end
    f:close()
    return videoList, cutNameList
end

function DataHandler:parse_json()
    local f = io.open(self.gtPath, 'r')
    for line in f:lines() do
        json_text = line
	break
    end
    local json = cjson.decode(json_text)
    return json["database"], json["version"], json["taxonomy"]
end

function DataHandler:create_dataset()
--  Get DataSet
    trainSet, valSet, testSet = {}, {}, {}
    for i = 1, #self.cutNameList do
        name = self.cutNameList[i]
        if not self.database[name] then
	    print("Video " .. name.. " not found in json")
	elseif self.database[name]['subset'] == "training" then
	    table.insert(trainSet, self.videoList[i])
	elseif self.database[name]['subset'] == "validation" then
	    table.insert(valSet, self.videoList[i])
	elseif self.database[name]['subset'] == 'testing' then
	    table.insert(testSet, self.videoList[i])
	end

        -- Get GT labels and segments
	local annotations = self.database[name]["annotations"]
	local duration = self.database[name]["duration"]
	local gt_segments = torch.Tensor(#annotations, 2)
	local gt_labels = torch.Tensor(#annotations)
	for j = 1, #annotations do
	    local segment = annotations[j]["segment"]
	    local st, ed = segment[1] / duration, segment[2] / duration
	    gt_segments[j] = torch.Tensor({st, ed})
	    local labelName = annotations[j]["label"]
	    gt_labels[j] = self.name2label[labelName]
	end
	self.database[name]["gt_segments"] = gt_segments
	self.database[name]["gt_labels"] = gt_labels
    end
    return trainSet, valSet, testSet

end

function DataHandler:get_classes()
    local name2label = {}
    local label2name = {}
    local cnt = 0
    for id, taxon in pairs(self.id2Tax) do
        if #taxon["child"] == 0 then
	    name2label[taxon["nodeName"]] = cnt
	    label2name[cnt] = taxon["nodeName"]
	    cnt = cnt + 1
	end
    end
    return name2label, label2name
end

function DataHandler:create_taxonomy_tree()
    self.id2Tax = {}  -- nodeId -> taxonomy
    self.subTree = {}  --> subtree rooted from nodeId
    for i = 1, #self.taxonomy do
        --local id = self.taxonomy[i]["nodeId"]
	local nodeId = self.taxonomy[i]["nodeId"]
	self.id2Tax[nodeId] = self.taxonomy[i]
	local parentId = self.taxonomy[i]["parentId"]
	if not self.subTree[parentId] then
	    self.subTree[parentId] = {}
	end
	table.insert(self.subTree[parentId], nodeId)
    end
    local root = self:recursive_contruct_tree(0)
    return root
end

function DataHandler:recursive_contruct_tree(rootid)
    local child = {}
    local annotation = self.id2Tax[rootid] -- note: id2Tax will be changed along with annotation( child added)
    local subroot = self.subTree[rootid]
    if not subroot then
        self.subTree[rootid] = {}
	annotation["child"] = child
	return annotation
    end
    for _, cid in pairs(subroot) do
        table.insert(child, self:recursive_contruct_tree(cid)) 
    end
    annotation["child"] = child
    return annotation
end

function DataHandler:cut_name(name)
    name = string.sub(name, 3, -1)
    return utils.split(name, ".")[1]
end
