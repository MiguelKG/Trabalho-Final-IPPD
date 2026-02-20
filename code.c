#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <omp.h>

#define MAX_AGENTS 1000
#define MAX_RESOURCES_BASIS 100
#define MAX_COST 1000000

int global_width = 10;
int global_height = 10;
int total_steps = 100;
int season_length = 5;
int total_agents = 10;

// Enums

typedef enum {
    VILLAGE,
    FISHING,
    GATHERING,
    FARMING,
    RESTRICTED
} TerrainType;

typedef enum {
    DRY,
    WET
} Season;

// Structs

typedef struct {
    TerrainType type;
    float resource;
    float consumption_pool;
    bool accessible;
} Cell;

typedef struct {
    int x, y;
    float energy;
    bool alive;
} Agent;

// Funções Auxiliares

TerrainType get_cell_type(int x, int y) {
    int value = (x + y) % 5;

    switch (value) {
        case 0: return VILLAGE;
        case 1: return FISHING;
        case 2: return GATHERING;
        case 3: return FARMING;
        default: return RESTRICTED;
    }
}

char* get_cell_type_text(TerrainType type) {
    switch (type) {
        case VILLAGE: return "Aldeia";
        case FISHING: return "Pesca";
        case GATHERING: return "Coleta";
        case FARMING: return "Rocado";
        default: return "Restrito";
    }
}

float get_initial_resource(TerrainType type) {
    switch (type) {
        case VILLAGE: return 20.0;
        case FISHING: return 30.0;
        case GATHERING: return 25.0;
        case FARMING: return 35.0;
        case RESTRICTED: return 0.0;
    }
}

bool is_accessible(TerrainType type, Season season) {

    if (type == RESTRICTED)
        return false;

    // Fazendas alagadas
    if (type == FARMING && season == WET)
        return false;

    // Rios secos
    if (type == FISHING && season == DRY)
        return false;

    return true;
}

float get_regeneration(TerrainType type, Season season) {

    switch (type) {

        case VILLAGE:
            return 3.0;

        case FISHING:
            return (season == WET) ? 5.0 : 1.0;

        case GATHERING:
            return 3.0;

        case FARMING:
            return (season == DRY) ? 10.0 : 0.0;

        case RESTRICTED:
            return 0.0;
    }

    return 0.0;
}

// Carga computacional sintética
void perform_synthetic_work(float cost) {

    long iterations = (long)(cost * 1000);
    if (iterations > MAX_COST) iterations = MAX_COST;

    double dummy = 0;
    for (long i = 0; i < iterations; i++) {
        dummy += i * 0.0001;
    }
}

// Decide movimento dos agentes
void decide_movement(Agent *agent, int *new_x, int *new_y,
    int width, int height, Season season,
    Cell **grid, float *h_left, float *h_right, unsigned int *seed) {

    float best_resource = -1.0;
    int best_x = agent->x;
    int best_y = agent->y;

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {

            int nx = agent->x + dx;
            int ny = agent->y + dy;

            if (ny < 0 || ny >= height) {
                continue;
            }

            float r = -1.0;
            if (nx >= 0 && nx < width) {
                if (grid[nx][ny].accessible) {
                    r = grid[nx][ny].resource;
                }
            } else {
                if (nx < 0) {
                    r = h_left[ny];
                } else
                if (nx >= width) {
                    r = h_right[ny];
                }
            }

            if (r > best_resource) {
                best_resource = r;
                best_x = nx;
                best_y = ny;
            } else
            if (
                r >= (best_resource - best_resource * 0.25) &&
                r > 0 &&
                rand_r(seed) % 101 < 50
            ) {
                best_x = nx;
                best_y = ny;
            }
        }
    }
    *new_x = best_x;
    *new_y = best_y;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    srand(42 + rank);
    int local_width = global_width / num_processes;
    int offset_x = rank * local_width;

    // Halos
    float *h_left = malloc(global_height * sizeof(float));
    float *h_right = malloc(global_height * sizeof(float));
    float *h_send = malloc(global_height * sizeof(float));

    // Inicializa Grid
    Cell** grid = malloc(local_width * sizeof(Cell*));
    for (int i = 0; i < local_width; i++) {
        grid[i] = malloc(global_height * sizeof(Cell));
    }

    Season season = DRY;

    for (int i = 0; i < local_width; i++) {
        for (int j = 0; j < global_height; j++) {
            TerrainType type = get_cell_type(offset_x + i, j);
            grid[i][j].type = type;
            grid[i][j].resource = get_initial_resource(type);
            grid[i][j].consumption_pool = 0;
            grid[i][j].accessible = is_accessible(type, season);
        }
    }

    // Inicializa Agentes
    Agent agents[MAX_AGENTS];
    int local_agent_count = total_agents / num_processes;

    for (int i = 0; i < local_agent_count; i++) {
        agents[i].x = rand() % local_width;
        agents[i].y = rand() % global_height;
        agents[i].energy = 50.0;
        agents[i].alive = true;
    }

    // Loop Principal
    for (int step = 1; step < total_steps; step++) {

        // Atualiza Estação
        if (step % season_length == 0 && step > 0) {
            season = (season == DRY) ? WET : DRY;
        }

        MPI_Bcast(&season, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Troca de Halos
        int left_neighbor = (rank - 1 + num_processes) % num_processes;
        int right_neighbor = (rank + 1) % num_processes;

        for(int j = 0; j < global_height; j++) {
            h_send[j] = grid[0][j].resource;
        }

        MPI_Sendrecv(h_send, global_height, MPI_FLOAT, left_neighbor, 10, h_right,
            global_height, MPI_FLOAT, right_neighbor, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(int j = 0; j < global_height; j++) {
            h_send[j] = grid[local_width - 1][j].resource;
        }
        
        MPI_Sendrecv(h_send, global_height, MPI_FLOAT, right_neighbor, 11, h_left,
            global_height, MPI_FLOAT, left_neighbor, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Processa Agentes
        #pragma omp parallel for
        for (int i = 0; i < local_agent_count; i++) {

            unsigned int seed = 42 + rank + omp_get_thread_num();

            Agent *agent = &agents[i];

            if (!agent->alive) continue;

            agent->energy -= (float) (10 + (rand_r(&seed) % 11));

            if (agent->energy <= 0) {
                agent->alive = false;
                continue;
            }

            perform_synthetic_work(grid[agent->x][agent->y].resource);

            int new_x, new_y;
            decide_movement(agent, &new_x, &new_y, local_width, global_height,
                season, grid, h_left, h_right, &seed);

            if (new_x >= 0 && new_x < local_width) {
                float want_to_consume = 25.0f;

                #pragma omp atomic
                grid[new_x][new_y].consumption_pool += want_to_consume;
                
                agent->x = new_x;
                agent->y = new_y;
                agent->energy += want_to_consume;
            } else {
                agent->x = new_x;
                agent->y = new_y;
            }
        }

        // Migração de Agentes (Sincronização MPI)

        // Buffers temporários para separar os agentes que saem e os que ficam
        Agent agents_to_send_left[MAX_AGENTS];
        Agent agents_to_send_right[MAX_AGENTS];
        Agent agents_staying_here[MAX_AGENTS];

        int count_to_left = 0;
        int count_to_right = 0;
        int count_staying = 0;

        // Identifica para onde cada agente deve ir baseado na sua coordenada X
        for (int i = 0; i < local_agent_count; i++) {
            if (agents[i].x < 0) {
                // Agente saiu pela esquerda: ajusta a posição para a borda do vizinho e adiciona à lista de envio
                agents[i].x = local_width - 1;
                agents_to_send_left[count_to_left++] = agents[i];
            } 
            else if (agents[i].x >= local_width) {
                // Agente saiu pela direita: ajusta a posição para o início do vizinho e adiciona à lista de envio
                agents[i].x = 0;
                agents_to_send_right[count_to_right++] = agents[i];
            }
            else {
                // Agente continua dentro do território deste processo
                agents_staying_here[count_staying++] = agents[i];
            }
        }

        // Variáveis para armazenar quantos agentes vamos receber dos vizinhos
        int incoming_from_right_count;
        int incoming_from_left_count;

        // Informa aos vizinhos quantos agentes estão a caminho para eles prepararem o espaço
        MPI_Sendrecv(&count_to_left, 1, MPI_INT, left_neighbor, 0, 
                    &incoming_from_right_count, 1, MPI_INT, right_neighbor, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&count_to_right, 1, MPI_INT, right_neighbor, 1, 
                    &incoming_from_left_count, 1, MPI_INT, left_neighbor, 1,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Buffer para armazenar todos os novos agentes recebidos de ambos os lados
        Agent incoming_agents_buffer[MAX_AGENTS];

        // Envia os agentes e recebe os novos
        // Recebendo da direita (o que foi enviado para a esquerda por ele)
        MPI_Sendrecv(agents_to_send_left, count_to_left * sizeof(Agent), MPI_BYTE, left_neighbor, 2, 
                    incoming_agents_buffer, incoming_from_right_count * sizeof(Agent), MPI_BYTE, right_neighbor, 2, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Recebendo da esquerda (o que foi enviado para a direita por ele)
        MPI_Sendrecv(agents_to_send_right, count_to_right * sizeof(Agent), MPI_BYTE, right_neighbor, 3, 
                    &incoming_agents_buffer[incoming_from_right_count], incoming_from_left_count * sizeof(Agent), MPI_BYTE, left_neighbor, 3, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 4. Reconstrução da lista local de agentes
        // Primeiro, recolocamos quem nunca saiu
        for (int i = 0; i < count_staying; i++) {
            agents[i] = agents_staying_here[i];
        }

        // Depois, adicionamos os imigrantes que acabaram de chegar
        int total_incoming = incoming_from_right_count + incoming_from_left_count;

        for (int i = 0; i < total_incoming; i++) {
            agents[count_staying + i] = incoming_agents_buffer[i];
        }

        // Atualiza o contador total de agentes sob responsabilidade deste processo
        local_agent_count = count_staying + total_incoming;

        // Atualizar Grid
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < local_width; i++) {
            for (int j = 0; j < global_height; j++) {

                grid[i][j].resource -= grid[i][j].consumption_pool;
                grid[i][j].resource += get_regeneration(grid[i][j].type, season);

                if (grid[i][j].resource < 0) {
                    grid[i][j].resource = 0;
                }

                if (grid[i][j].resource > (get_regeneration(grid[i][j].type, season) / 10) * MAX_RESOURCES_BASIS) {
                    grid[i][j].resource = (get_regeneration(grid[i][j].type, season) / 10) * MAX_RESOURCES_BASIS;
                }

                grid[i][j].consumption_pool = 0;
                grid[i][j].accessible = is_accessible(grid[i][j].type, season);
            }
        }

        float local_energy_sum = 0;

        for (int i = 0; i < local_agent_count; i++) {
            local_energy_sum += agents[i].energy;
        }

        float global_energy_sum;

        MPI_Allreduce(&local_energy_sum, &global_energy_sum,
                      1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Step %d - Total Energy: %.2f\n", step, global_energy_sum);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // --Exibição dos Resultados
    for (int p = 0; p < num_processes; p++) {
        if (rank == p) {
            FILE *f = fopen("output.txt", (rank == 0) ? "w" : "a");
            if (!f) { printf("Erro ao abrir arquivo no rank %d\n", rank); }

            fprintf(f, "\n========================================");
            fprintf(f, "\nPROCESSOR %d | Agentes ativos: %d\n", rank, local_agent_count);
            fprintf(f, "========================================\n");

            for (int i = 0; i < local_agent_count; i++) {
                fprintf(f, "Agente %d -> Pos(%d,%d) | Energia: %.2f | Status: %s\n",
                    i, agents[i].x, agents[i].y, agents[i].energy, 
                    agents[i].alive ? "VIVO" : "MORTO");
            }

            for (int i = 0; i < local_width; i++) {
                for (int j = 0; j < global_height; j++) {
                    int type = grid[i][j].type;
                    fprintf(f, "(%-2d, %-2d) %-9s; recurso = %5.2f; agentes = \n", offset_x + i, j,
                        get_cell_type_text(grid[i][j].type), grid[i][j].resource);
                    
                    for (int k = 0; k < local_agent_count; k++) {
                        if (agents[k].x == i && agents[k].y == j) {
                            fprintf(f, "Agente %d (%c) (%.2f)\n", k, agents[k].alive ? 'V' : 'M', agents[k].energy);
                        }
                    }
                }
            }
            fflush(stdout);
            fclose(f);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // -------

    for (int i = 0; i < local_width; i++) free(grid[i]);
    free(grid);
    free(h_left); free(h_right); free(h_send);

    MPI_Finalize();
    return 0;
}