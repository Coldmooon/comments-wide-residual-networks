local utils = {}

function utils.MSRinit(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nInputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
end

function utils.BilinearFiller(model)
   for k,v in pairs(model:findModules('nn.SpatialFullConvolution')) do
      local f = math.ceil (v.kW / 2)
      local c = (2 * f - 1 - math.fmod(f,2)) / ( 2 * f)
      local count = v.weight:size()[1]*v.weight:size()[2]*v.weight:size()[3]*v.weight:size()[4]
      v.weight:permute(2,1,3,4)
      for dim = 1,count do
         local x = math.fmod(dim, v.kW)
         local y = math.fmod((dim / v.kW), v.kH)
         -- print('dim: ', dim)
         -- print('value: ', (1 - math.abs( x / f - c)) * (1 - math.abs(y / f - c)))
         v.weight:storage()[dim] = (1 - math.abs( x / f - c)) * (1 - math.abs(y / f - c))
         -- local index = {}
         -- index[1] = math.floor( (dim -1) / (v.kW*v.kH*v.nInputPlane)) + 1
         -- index[2] = math.floor( math.fmod( dim - 1, (v.kW*v.kH*v.nInputPlane)) / (v.kW*v.kH) ) + 1
         -- index[4] = math.fmod( dim - 1, v.kH) + 1
         -- index[3] = math.fmod( dim - 1, v.kW) + 1
         -- v.weight[index[1]][index[2]][index[3]][index[4]] = value
         -- v.weight:storage()[dim] = value
      end
      v.weight:permute(2,1,3,4)
      if v.bias then v.bias:zero() end
end
end

function utils.FCinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.bias:zero()
   end
end

function utils.DisableBias(model)
   for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
      v.bias = nil
      v.gradBias = nil
   end
end

function utils.testModel(model)
   model:float()
   local imageSize = opt and opt.imageSize or 32
   local input = torch.randn(1,3,imageSize,imageSize):type(model._type)
   print('forward output',{model:forward(input)})
   print('backward output',{model:backward(input,model.output)})
   model:reset()
end

function utils.makeDataParallelTable(model, nGPU)
   if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end
   return model
end

return utils
