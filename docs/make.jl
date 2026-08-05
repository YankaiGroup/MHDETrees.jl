using MHDETrees
using Documenter

const PACKAGE_AUTHORS = "Jiayang Ren, Valentín Osuna-Enciso, and Yankai Cao"

makedocs(
    modules=[MHDETrees],
    authors=PACKAGE_AUTHORS,
    sitename="MHDETrees.jl",
    pages=["Home" => "index.md"],
    checkdocs=:exports,
    remotes=nothing,
    format=Documenter.HTML(edit_link=nothing, repolink=nothing),
)
