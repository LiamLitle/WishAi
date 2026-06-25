# estimations.jl
# Ce script tourne en arrière-plan, lit les logs de l'IA et effectue des prédictions mathématiques.

using JSON
using LsqFit

# Le chemin racine du projet (suppose que le script est dans src/)
const ROOT_DIR = dirname(@__DIR__)
const MODEL_DIR = joinpath(ROOT_DIR, "model")
const ACTIVE_FILE = joinpath(MODEL_DIR, "active.json")

function get_active_log_file()
    if !isfile(ACTIVE_FILE)
        return nothing
    end
    try
        data = JSON.parsefile(ACTIVE_FILE)
        model_name = get(data, "model", nothing)
        if model_name !== nothing
            return joinpath(MODEL_DIR, model_name, "log_active.json")
        end
    catch
        return nothing
    end
    return nothing
end

# Formule exponentielle: y = A * exp(-B*x) + C
# où C est le plateau (valeur minimale asymptotique)
@. model_exp(x, p) = p[1] * exp(-p[2] * x) + p[3]

function check_chinchilla(nb_params, nb_tokens)
    ratio = nb_tokens / nb_params
    risque = "faible"
    etape_estimee = -1
    
    if ratio >= 20
        risque = "faible"
    elseif ratio >= 10
        risque = "modéré"
        etape_estimee = Int(round(nb_tokens / 1000)) # Heuristique grossière
    elseif ratio >= 5
        risque = "élevé"
        etape_estimee = Int(round(nb_tokens / 2000))
    else
        risque = "critique"
        etape_estimee = Int(round(nb_tokens / 5000))
    end
    
    return Dict(
        "ratio" => round(ratio, digits=2),
        "risque" => risque,
        "etape_overfitting_estimee" => etape_estimee > 0 ? etape_estimee : "n/a"
    )
end

function estimate_plateau(steps)
    if length(steps) < 5
        return Dict("status" => "insuffisant", "plateau" => nothing, "etapes_restantes" => nothing)
    end
    
    val_losses = Float64[]
    iterations = Float64[]
    
    for (i, step) in enumerate(steps)
        val_loss = get(step, "val_loss", nothing)
        iter = get(step, "step", nothing)
        if val_loss !== nothing && iter !== nothing
            push!(val_losses, Float64(val_loss))
            push!(iterations, Float64(iter))
        end
    end
    
    if length(val_losses) < 5
        return Dict("status" => "insuffisant", "plateau" => nothing, "etapes_restantes" => nothing)
    end
    
    # Pour le fitting, on utilise l'index plutôt que l'itération brute pour stabiliser LsqFit
    x_data = Float64.(1:length(val_losses))
    y_data = val_losses
    
    # Paramètres initiaux: A = max - min, B = 0.1, C = min
    p0 = [maximum(y_data) - minimum(y_data), 0.1, minimum(y_data)]
    
    try
        fit = curve_fit(model_exp, x_data, y_data, p0)
        p = fit.param
        
        A, B, C = p
        
        # C est le plateau estimé
        plateau_estime = round(C, digits=4)
        
        # S'assurer que le plateau est un minimum logique
        if plateau_estime > minimum(y_data) || plateau_estime < 0
            plateau_estime = round(minimum(y_data) - 0.05, digits=4)
            plateau_estime = max(plateau_estime, 0.5) # Hard minimum
        end
        
        val_actuel = y_data[end]
        
        etapes_restantes = 0
        confiance = "faible"
        
        if B > 0 && val_actuel > plateau_estime + 0.05
            # Combien d'évaluations restantes pour atteindre plateau + 0.05
            # val_actuel = A * exp(-B * x_current) + C
            # cible = plateau_estime + 0.05
            # -> on cherche Δx
            delta_x = -log(0.05 / (val_actuel - plateau_estime)) / B
            if delta_x > 0
                # Convertir delta_x (nombre d'évaluations) en étapes réelles
                step_interval = length(iterations) > 1 ? (iterations[end] - iterations[end-1]) : 500
                etapes_restantes = Int(round(delta_x * step_interval))
            end
        end
        
        if length(y_data) >= 10
            confiance = "bonne"
        elseif length(y_data) >= 5
            confiance = "moyenne"
        end
        
        return Dict(
            "status" => "succès",
            "plateau" => plateau_estime,
            "etapes_restantes" => etapes_restantes > 0 ? etapes_restantes : 0,
            "confiance" => confiance
        )
    catch e
        return Dict("status" => "erreur", "message" => string(e), "plateau" => nothing)
    end
end

function main_loop()
    println("🔮 Julia Analytics Engine démarré. En attente de données...")
    
    while true
        sleep(5)
        
        log_file = get_active_log_file()
        if log_file === nothing || !isfile(log_file)
            continue
        end
        
        try
            log_data = JSON.parsefile(log_file)
            hyperparams = get(log_data, "hyperparams", Dict())
            steps = get(log_data, "steps", [])
            
            insights = Dict{String, Any}()
            
            # 1. Chinchilla Check
            nb_params = get(hyperparams, "nb_params", nothing)
            # Estimation du dataset (peut être manquant dans les logs, utilisons un dummy si besoin,
            # mais idéalement on devrait le logger. Dans Python on l'ajoutera).
            # On va utiliser une variable passée dans hyperparams.
            nb_tokens = get(hyperparams, "nb_tokens_train", nothing)
            
            if nb_params !== nothing && nb_tokens !== nothing
                insights["chinchilla"] = check_chinchilla(nb_params, nb_tokens)
            else
                insights["chinchilla"] = Dict("status" => "en attente de données (nb_tokens_train)")
            end
            
            # 2. Estimation du Plateau
            insights["convergence"] = estimate_plateau(steps)
            
            # Sauvegarder dans insights.json
            insights_file = joinpath(dirname(log_file), "insights.json")
            open(insights_file, "w") do io
                JSON.print(io, insights, 4)
            end
            
        catch e
            # Ignorer les erreurs de parsing partielles
            # println("Erreur de lecture: ", e)
        end
    end
end

main_loop()
