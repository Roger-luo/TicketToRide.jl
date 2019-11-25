export NoGate

using YaoBase, YaoArrayRegister, YaoBlocks

struct NoGate <: PrimitiveBlock{1} end

YaoBlocks.apply!(r::AbstractRegister, x::NoGate) = r
YaoBase.instruct!(r::ArrayReg, x::NoGate, locs) = r
YaoBlocks.mat_matchreg(r::ArrayReg, x::NoGate) = x