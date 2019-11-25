using YaoBase, YaoArrayRegister, YaoBlocks
export Depolar
struct Depolar <: PrimitiveBlock{1}
    ps::NTuple{4, Float64}

    function Depolar(ps::NTuple{4, Float64})
        Z = sum(ps)
        if !isone(Z)
            new(map(x->x/Z, ps))
        else
            new(ps)
        end
    end
end

Depolar(ps...) = Depolar(ps)
Depolar() = Depolar(ntuple(_->0.5, 4))

function YaoBlocks.apply!(r::AbstractRegister, x::Depolar)
    instruct!(r, x, (1, ))
end

YaoBlocks.mat_matchreg(::AbstractRegister, x::Depolar) = x
YaoBlocks.mat_matchreg(::ArrayReg, x::Depolar) = x

function YaoBase.instruct!(r::ArrayReg, x::Depolar, locs::Tuple)
    _instruct!(r, x, locs)
end

function YaoBase.instruct!(r::AbstractRegister, x::Depolar, locs::Tuple)
    _instruct!(r, x, locs)
end

function _instruct!(r::AbstractRegister, x::Depolar, locs::Tuple)
    dice = rand()
    if dice < x.ps[1]
        instruct!(r, Val(:X), locs)
    elseif dice < sum(x.ps[1:2])
        instruct!(r, Val(:Y), locs)
    elseif dice < sum(x.ps[1:3])
        instruct!(r, Val(:Z), locs)
    end
    return r
end