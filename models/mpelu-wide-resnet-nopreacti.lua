--  Wide Residual Network
--  This is an implementation of the wide residual networks described in:
--  "Wide Residual Networks", http://arxiv.org/abs/1605.07146
--  authored by Sergey Zagoruyko and Nikos Komodakis

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

require './mpelu/fast-mpelu'

local nn = require 'nn'
local utils = paths.dofile'utils.lua'

-- Just for convenience.
local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local alpha = 1
local beta = 1

local function createModel(opt)
   assert(opt and opt.depth)
   assert(opt and opt.num_classes)
   assert(opt and opt.widen_factor)

   local function Dropout()
      return nn.Dropout(opt and opt.dropout or 0,nil,true)
   end

   local depth = opt.depth

   local blocks = {}
   
   -- This function is used to creat a residual modual.
   local function wide_basic(nInputPlane, nOutputPlane, stride)
      -- In each residual modual, There are two convolutional layers;
      -- For the first one, the stride of conv should be determined by the
      -- type of moudal. If the modual is used to downsample, stride = 2;
      -- stride = 1, otherwise. For the second one, it is corresponding to
      -- the second convoutional layer whose parameters are never changed.
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane

      local block = nn.Sequential() -- block is the modual container.
      local convs = nn.Sequential() -- convs is the residual branch container.    

      -- Pipeline:
      -- make sure the type of the module;
      -- put `BatchNorm` and `ReLU` somewhere according to the type;
      -- add the first conv and set `stride` according to the type;
      -- add `BatchNorm` and `ReLU` again;
      -- add the second conv.
      for i,v in ipairs(conv_params) do
         if i == 1 then  -- in which case, v == {3,3,stride,stride,1,1}
            -- This line is used to confirm the type of the module, a regular one or
            -- a downsample one.
            local module = nInputPlane == nOutputPlane and convs or block
            -- For the first convolutional layer in the modual, if nInputPlane == nOutputPlane
            -- , the type of modual is just a regular one. In this case,
            -- module == convs. Therefore, put `BatchNorm` and `ReLU` in the residual branch.
            -- If nInputPlane != nOutputPlane, the type is a downsample modual. In this case,
            -- module == block and put `BatchNorm` and `ReLU` out of the module. All in all, 
            -- the next line is only used to decide where to put `BatchNorm` and `ReLU`.
            
            -- module:add(SBatchNorm(nInputPlane ,false)):add(MPELU(alpha, beta, 5, 5, 5, 5, nInputPlane))
            -- The first convolutional layer, its `stride` depends on the type of the module.
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
         else -- in which case, v == {3,3,1,1,1,1}
            -- First add `BatchNorm` and `ReLU`.
            convs:add(SBatchNorm(nBottleneckPlane, false)):add(MPELU(alpha, beta, 3, 3, 0, 0, nBottleneckPlane))
            if opt.dropout > 0 then
               convs:add(Dropout())
            end
            -- Then add the second convolutional layer into the module.
            convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
            convs:add(SBatchNorm(nBottleneckPlane, false))
         end
      end
      -- Add shorcut branch for the module.
      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         nn.Sequential():add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)):add(SBatchNorm(nOutputPlane, false))
     
      -- Aggregate the two branches by the operation of `ADD`.
      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   do
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6

      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(SBatchNorm(nStages[1], false))
      model:add(MPELU(alpha, beta, 3, 3, 0, 0, nStages[1]))
      model:add(layer(wide_basic, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(wide_basic, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4], false))
      model:add(MPELU(alpha, beta, 3, 3, 0, 0, nStages[4]))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], opt.num_classes))
   end

   utils.DisableBias(model)
   utils.testModel(model)
   utils.MSRinit(model)
   utils.FCinit(model)
   
   -- model:get(1).gradInput = nil
   print('Parameters', model:getParameters():size()[1])
   return model
end

return createModel
