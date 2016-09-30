-------------------j-----------------------------
-- utility functions for the evaluation part
------------------------------------------------
utils = {}
function utils.interval_overlap(gts, dets)
  local num_gt = gts:size(1)
  local num_det = dets:size(1)
  local ov = torch.Tensor(num_gt, num_det)
  for i=1,num_gt do
    for j=1,num_det do
      ov[i][j] = utils.interval_overlap_single(gts[i], dets[j])
    end
  end
  return ov
end

function utils.interval_overlap_single(gt, dt)
  local i1 = gt
  local i2 = dt
  -- union
  local bu = {math.min(i1[1], i2[1]), math.max(i1[2], i2[2])}
  local ua = bu[2] - bu[1]
  -- overlap
  local ov = 0
  local bi = {math.max(i1[1], i2[1]), math.min(i1[2], i2[2])}
  local iw = bi[2] - bi[1]
  if iw > 0 then
    ov = iw / ua
  end
  return ov
end

function utils.round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end

function utils.get_gts(input, target)
    local batch_size = #target
    local gt_set = torch.Tensor(batch_size, 2)
    for i = 1, batch_size do
        local det = torch.Tensor(2)
	det:copy(input[i]):resize(1, 2)
	local gts = target[i]
	local overlaps = utils.interval_overlap(gts, det)
	local max_ov, max_ovid  = torch.max(overlaps, 1)
	if (max_ov[1][1] > 0) then
	    gt_set[i] = gts[max_ovid[1][1]]
	else
	    local dists = utils.segment_distance(gts, det)    
	    local min_dis, min_disid = torch.min(dists, 1)
	    gt_set[i] = gts[min_disid[1][1]]
	end
    end
    return gt_set
end

function utils.segment_distance(gts, dets)
    local num_gt = gts:size(1)
    local num_det = dets:size(1)
    local dists = torch.Tensor(num_gt, num_det)
    for i=1,num_gt do
        for j=1,num_det do
            dists[i][j] = utils.segment_distance_single(gts[i], dets[j])
        end
    end
    return dists
end

function utils.segment_distance_single(gt, det)
    local s1, e1 = gt[1], gt[2]
    local s2, e2 = det[1], det[2]
    if (s1 > s2) then
        return math.max(s1 - e2, 0)
    else
        return math.max(s2 - e1, 0)
    end
end

function utils.split(s, splitor)
    local t = {}
    local regexp = "([^'" .. splitor .. "']+)"
    for w in string.gmatch(s, regexp) do 
        table.insert(t, w)
    end
    return t
end

return utils
