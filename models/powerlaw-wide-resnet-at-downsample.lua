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
require 'nnlr'
local nn = require 'nn'
local utils = paths.dofile'utils.lua'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   assert(opt and opt.depth)
   assert(opt and opt.num_classes)
   assert(opt and opt.widen_factor)

   local function Dropout()
      return nn.Dropout(opt and opt.dropout or 0,nil,true)
   end

   local depth = opt.depth

   local blocks = {}
   
   local function wide_basic(nInputPlane, nOutputPlane, stride)
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane

      local block = nn.Sequential()
      local convs = nn.Sequential()     

      -- residual branch
      for i,v in ipairs(conv_params) do      	 
         if i == 1 then -- when i = 1, v = {3,3,stride,stride,1,1}
         	-- check if nInputPlane is equal to nOutputPlane
            local module = nInputPlane == nOutputPlane and convs or block
            -- module:add(SBatchNorm(nInputPlane)):add(ReLU(true)) -- first add BN, then ReLU
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
         else -- when i = 2, v = {3,3,1,1,1,1}
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(true))
            if opt.dropout > 0 then
               convs:add(Dropout())
            end
            convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
         end
      end
     
      -- shorcut branch
      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)
     
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

   -- make power law block
--   local powerlaw_end = nn.Concat(3) 
--   
--   powerlaw_end:add(Avg(8, 8, 1, 1))
--   
--   downto7 = nn.Sequential()
--   scale7 = nn.CMul(640,7,7)
--   scale7.weight = torch.Tensor(640,7,7):fill(1.0391)
--   downto7:add(Max(2, 2, 1, 1)):add(scale7):add(Avg(7, 7, 1, 1))
--
--   downto6 = nn.Sequential()
--   scale6 = nn.CMul(640,6,6)
--   scale6.weight = torch.Tensor(640,6,6):fill(1.0874)
--   downto6:add(Max(3, 3, 1, 1)):add(scale6):add(Avg(6, 6, 1, 1))
--
--   downto5 = nn.Sequential()
--   scale5 = nn.CMul(640,5,5)
--   scale5.weight = torch.Tensor(640,5,5):fill(1.1495)
--   downto5:add(Max(2, 2, 2, 2, 1, 1)):add(scale5):add(Avg(5, 5, 1, 1))
--
--   powerlaw_end:add(downto7):add(downto6):add(downto5)

   ------------------------------------------------------

   local powerlaw32to16 = nn.Concat(2)
   -- powerlaw32to16:add(nn.Identity())
   powerlaw32to16:add(nn.Sequential():add(Convolution(160, 40, 1, 1, 1, 1, 0, 0)):add(SBatchNorm(40)):add(ReLU(true)))

   deconv64 = nn.Sequential()
   deconv64:add(nn.SpatialUpSamplingNearest(2))
   -- deconv64:add(nn.SpatialFullConvolution(160, 160, 4, 4, 2, 2, 1, 1):noBias():learningRate('weight', 0):weightDecay('weight', 0))
   -- scale64 = nn.CMul(160, 1, 1)
   -- scale64.weight = torch.Tensor(160, 1, 1):fill(1.0391)
   scale64 = nn.Mul()
   -- scale64.weight = 1.0391
   deconv64:add(scale64)
   deconv64:add(Convolution(160, 40, 4, 4, 2, 2, 1, 1))
   deconv64:add(SBatchNorm(40)):add(ReLU(true))

   deconv96 = nn.Sequential()
   deconv96:add(nn.SpatialUpSamplingNearest(3))
   --deconv96:add(nn.SpatialFullConvolution(160, 160, 5, 5, 3, 3, 1, 1):noBias():learningRate('weight', 0):weightDecay('weight', 0))
   -- scale96 = nn.CMul(160, 1, 1)
   -- scale96.weight = torch.Tensor(160, 1, 1):fill(1.0874)
   scale96 = nn.Mul()
   -- scale96.weight = 1.0874
   deconv96:add(scale96)
   deconv96:add(Convolution(160, 40, 5, 5, 3, 3, 1, 1))
   deconv96:add(SBatchNorm(40)):add(ReLU(true))

   deconv128 = nn.Sequential()
   deconv128:add(nn.SpatialUpSamplingNearest(4))
   -- deconv128:add(nn.SpatialFullConvolution(160, 160, 8, 8, 4, 4, 2, 2):noBias():learningRate('weight', 0):weightDecay('weight', 0))
   -- scale128 = nn.CMul(160, 1, 1)
   -- scale128.weight = torch.Tensor(160, 1, 1):fill(1.1495)
   scale128 = nn.Mul()
   -- scale128.weight = 1.1495
   deconv128:add(scale128)
   deconv128:add(Convolution(160, 40, 8, 8, 4, 4, 2, 2))
   deconv128:add(SBatchNorm(40)):add(ReLU(true))

   powerlaw32to16:add(deconv64):add(deconv96):add(deconv128)
   ---------------------------------------------------------
   -- local powerlaw16to8 = nn.ConcatTable(3)
   -- powerlaw16to8:add(nn.Identity())

   -- deconv64 = nn.Sequential()
   -- deconv64:add(nn.SpatialFullConvolution(320, 320, 4, 4, 2, 2, 1, 1):noBias())
   -- scale64 = nn.CMul(320, 32, 32)
   -- scale64.weight = torch.Tensor(320, 32, 32):fill(1.0391)
   -- deconv64:add(scale64)
   -- deconv64:add(Convolution(320, 320, 4, 4, 2, 2, 1, 1))
   -- deconv64:add(SBatchNorm(320)):add(ReLU(true))

   -- deconv96 = nn.Sequential()
   -- deconv96:add(nn.SpatialFullConvolution(320, 320, 5, 5, 3, 3, 1, 1):noBias())
   -- scale96 = nn.CMul(320, 48, 48)
   -- scale96.weight = torch.Tensor(320, 48, 48):fill(1.0874)
   -- deconv96:add(scale96)
   -- deconv96:add(Convolution(320, 320, 5, 5, 3, 3, 1, 1))
   -- deconv96:add(SBatchNorm(320)):add(ReLU(true))

   -- deconv128 = nn.Sequential()
   -- deconv128:add(nn.SpatialFullConvolution(320, 320, 8, 8, 4, 4, 2, 2):noBias())
   -- scale128 = nn.CMul(320, 64, 64)
   -- scale128.weight = torch.Tensor(320, 64, 64):fill(1.1495)
   -- deconv128:add(scale128)
   -- deconv128:add(Convolution(320, 320, 8, 8, 4, 4, 2, 2))
   -- deconv128:add(SBatchNorm(320)):add(ReLU(true))

   -- powerlaw16to8:add(deconv64):add(deconv96):add(deconv128)

   local model = nn.Sequential()
   do
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6

      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(SBatchNorm(nStages[1]))
      model:add(ReLU(true))
      model:add(layer(wide_basic, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      -- model:get(4):get(4):get(1):get(1):get(4).nOutputPlane = 40
      model:add(powerlaw32to16)--:add(nn.CAddTable())
      
      model:add(layer(wide_basic, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      -- model:add(powerlaw16to8):add(nn.CAddTable())

      model:add(layer(wide_basic, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(true))
      -- model:add(nn.Concat(3):add(Avg(8,8,1,1)):add(Avg(8,8,1,1)):add(Avg(8,8,1,1)))
      -- model:add(powerlaw_end)
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
