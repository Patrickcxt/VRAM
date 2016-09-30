require 'utils'
local cjson = require 'cjson'
require 'image'

local DataHandler = torch.class("DataHandler")

function DataHandler:__init()
    self.root = "/home/t5/cxt/VRAM/data/ActivityNet/"
    self.indexPath = self.root .. "scheme/index.txt"
    self.gtPath = self.root .. "scheme/activity_net.v1-3.min.json"
    self.videoList, self.cutNameList = self:create_video_list()
    self.database, self.version, self.taxonomy = self:parse_json()
    self.trainSet, self.valSet, self.testSet = self:create_dataset()
    print(self.database["YEZrwxz0Ysk"])
    self.root = self:create_taxonomy_tree()
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
    end
    return trainSet, valSet, testSet

    -- Refine Database


end

function DataHandler:create_taxonomy_tree()
    -- Construct taxonomy tree
    return nil
end

function DataHandler:cut_name(name)
    name = string.sub(name, 3, -1)
    return utils.split(name, ".")[1]
end
