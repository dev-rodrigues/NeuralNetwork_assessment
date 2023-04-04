import numpy as np


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Defina o número de nós nas camadas de entrada, ocultas e de saída.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Inicializa pesos
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5, (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5, (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
    
    def sigmoid(x):
        """
        Função sigmoid utilizada como função de ativação da rede neural.
        """
        return 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        """
        Função que treina a rede neural com os dados de entrada.

        Argumentos:
        - features: Matriz de características dos dados de treinamento, onde cada linha representa uma instância
        e cada coluna representa uma característica.
        - targets: Vetor com as classes dos dados de treinamento.

        Saída:
        Nenhum.

        Realiza atualizacao da rede neural com base nos dados de entrada e saída.
        """
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for features, target in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(features)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs, hidden_outputs, features, target, delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        """
            Função que executa a etapa de forward pass durante o treinamento da rede neural.

            Argumentos:
                - X: Um vetor de características de entrada.

            Saída:
                - final_outputs: A saída da rede neural após a camada de saída.
                - hidden_outputs: A saída da rede neural após a camada oculta.
            
            Essa função recebe um vetor de características de entrada e executa o forward pass na rede neural.
        """
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        """
            Função que executa a etapa de backpropagation durante o treinamento da rede neural.
            Argumentos:
                - final_outputs: A saída da rede neural após a camada de saída.
                - hidden_outputs: A saída da rede neural após a camada oculta.
                - X: Um vetor de características de entrada.
                - y: O valor alvo correspondente à entrada X.
                - delta_weights_i_h: O delta dos pesos da camada de entrada para a camada oculta.
                - delta_weights_h_o: O delta dos pesos da camada oculta para a camada de saída.

            Saída:
                - delta_weights_i_h: O delta atualizado dos pesos da camada de entrada para a camada oculta.
                - delta_weights_h_o: O delta atualizado dos pesos da camada oculta para a camada de saída.
            
            executa o backpropagation na rede neural e retorna os deltas atualizados dos pesos da camada de 
            entrada para a camada oculta e da camada oculta para a camada de saída.
        """
        error = y - final_outputs  # Erro = saída desejada - saída real
        output_error_term = error.reshape((1, error.shape[0]))

        hidden_error = np.dot(self.weights_hidden_to_output, error)
        hidden_error_term = hidden_error * (1 - hidden_outputs) * hidden_outputs

        delta_weights_i_h += hidden_error_term * X[:, None]
        
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Atualize os pesos na etapa de gradiente descendente

            Arguments
            ---------
            delta_weights_i_h: alteração nos pesos da entrada para as camadas ocultas
            delta_weights_h_o: alteração nos pesos das camadas ocultas para as de saída
            n_records: número de registros

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records  # atualize os pesos da camada oculta para a de saída com passo de gradiente descendente
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records  # atualize os pesos da camada de entrada para a oculta com passo de gradiente descendente

    def run(self, features):
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs # Como o valor de ativação da camada de saída é linear

        return final_outputs

