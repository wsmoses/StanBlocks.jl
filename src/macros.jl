broadcastable(x) = false # avoid dotting spliced objects (e.g. view calls inserted by @view)
# don't add dots to dot operators
broadcastable(x::Symbol) = (!Base.isoperator(x) || first(string(x)) != '.' || x === :..) && x !== :(:)
broadcastable(x::Expr) = x.head !== :$
unbroadcast(x) = x
function unbroadcast(x::Expr)
    if x.head === :.=
        Expr(:(=), x.args...)
    elseif x.head === :block # occurs in for x=..., y=...
        Expr(:block, Base.mapany(unbroadcast, x.args)...)
    else
        x
    end
end
__broadcasted__(x) = x
function __broadcasted__(x::Expr)
    broadcasted = :(Base.broadcasted)
    broadcastargs = Base.mapany(__broadcasted__, x.args)
    return if x.head === :call && broadcastable(x.args[1])
        Expr(:call, broadcasted, broadcastargs...)
    elseif x.head === :comparison
        error()
        Expr(:comparison, (iseven(i) && broadcastable(arg) && arg isa Symbol && Base.isoperator(arg) ?
                               Symbol('.', arg) : arg for (i, arg) in pairs(broadcastargs))...)
    elseif x.head === :$
        x.args[1]
    elseif x.head === :let # don't add dots to `let x=...` assignments
        Expr(:let, unbroadcast(broadcastargs[1]), broadcastargs[2])
    elseif x.head === :for # don't add dots to for x=... assignments
        Expr(:for, unbroadcast(broadcastargs[1]), broadcastargs[2])
    elseif (x.head === :(=) || x.head === :function || x.head === :macro) &&
           Meta.isexpr(x.args[1], :call) # function or macro definition
        Expr(x.head, x.args[1], broadcastargs[2])
    elseif x.head === :(<:) || x.head === :(>:)
        Expr(:call, broadcasted, x.head, broadcastargs...)
    else
        head = String(x.head)::String
        if last(head) == '=' && first(head) != '.' || head == "&&" || head == "||"
            Expr(:call, broadcasted, x.head, broadcastargs...)
        else
            Expr(x.head, broadcastargs...)
        end
    end
end
macro broadcasted(x)
    esc(__broadcasted__(x))
end


const X_NAME = gensym("x")
const TMP = gensym("tmp")
const XPOS = gensym("xpos")

macro_stan(x) = begin
    fname = gensym("stan_lpdf") 
    quote 
        function $fname($X_NAME) 
            target = 0.
            $x
            return target
        end
    end
end
macro stan(x)
    esc(macro_stan(x))
end
function macro_parameters end
macro parameters(block)
    esc(macro_parameters(block))
end
macro transformed_parameters(block)
    esc(block)
end
function macro_model end
macro model(block)
    esc(macro_model(block))
end
function macro_generated_quantities end
macro generated_quantities(block)
    esc(macro_generated_quantities(block))
end
begin
    macro_parameters(e) = e
    # macro_parameters(e::Symbol; xpos) = macro_parameters(:($e::real); xpos)
    # parameters_info(e::Symbol; xpos) = begin 
    #     @assert e in (:real, :vector) e
    #     :($X_NAME[$xpos]), xpos, -Inf, +Inf
    # end
    # parameters_info(e::Expr; xpos) = if e.head == :call
    #     X_VIEW, xslice, lower, upper = parameters_info(e.args[1]; xpos)
    #     for arg in e.args[2:end]
    #         @assert Meta.isexpr(arg, :kw) arg
    #         lower = arg.args[1] == :lower ? arg.args[2] : lower
    #         upper = arg.args[1] == :upper ? arg.args[2] : upper
    #     end
    #     X_VIEW, xslice, lower, upper
    # elseif e.head == :ref
    #     @assert length(e.args) == 2 e.args
    #     xslice = :($xpos:($xpos+$(e.args[2])-1))
    #     _, _, lower, upper = parameters_info(e.args[1]; xpos)
    #     :(view($X_NAME, $xslice)), xslice, lower, upper
    # else
    #     @error e.head
    #     error()
    # end
    extract_kws(e::Symbol) = e, ()
    extract_kws(e::Expr) = begin 
        @assert e.head == :call
        e.args[1]::Symbol
        @assert all([Meta.isexpr(arg, :kw) for arg in e.args[2:end]])
        e.args[1], e.args[2:end]
    end
    macro_parameters(e::Expr) = begin 
        if e.head == :block
            Expr(:block, :($XPOS=1), macro_parameters.(e.args)...)
        else
            @assert e.head == :(::)
            name, varinfo = e.args
            stmts = if !Meta.isexpr(varinfo, :ref)
                type, kws = extract_kws(varinfo)
                @assert type == :real
                [
                    :(($TMP, $name) = StanBlocks.constrain($X_NAME[$XPOS]; $(kws...))),
                    :(target += $TMP),
                    :($XPOS += 1)
                ]
            else
                type, kws = extract_kws(varinfo.args[1])
                dims = varinfo.args[2:end]
                constrain = Symbol(type, "_constrain")
                isdefined(StanBlocks, constrain) && (constrain = :(StanBlocks.$constrain))
                dim = Symbol(type, "_unconstrained_dim")
                isdefined(StanBlocks, dim) && (dim = :(StanBlocks.$dim))
                [
                    :($XPOS = $XPOS:($XPOS+$dim($(dims...))-1)),
                    :(($TMP, $name) = $constrain(view($X_NAME, $XPOS), $(dims...); $(kws...))),
                    :(target += $TMP),
                    :($XPOS = $XPOS[end]+1)
                ]
            end
            Expr(:block, stmts...)
        end
    end
    macro_model(e) = e
    macro_model(e::Expr) = begin 
        if e.head == :call && e.args[1] == :(~)
            lhs = e.args[2]
            rhs = e.args[3]
            @assert Meta.isexpr(rhs, :call)
            lpdf = Symbol("$(rhs.args[1])_lpdf")
            isdefined(StanBlocks, lpdf) && (lpdf = :(StanBlocks.$lpdf))
            rhsargs = rhs.args[2:end]
            rv = :(target += $lpdf($lhs, $(rhsargs...)))
            # display(Pair(e, rv))
            rv
        else
            Expr(macro_model(e.head), macro_model.(e.args)...)
        end
    end
end