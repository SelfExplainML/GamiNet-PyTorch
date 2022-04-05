import torch


class TensorLayer(torch.nn.Module):

    def __init__(self, n_subnets, subnet_arch, n_input_nodes, activation_func, device):
        super().__init__()

        self.device = device
        self.n_subnets = n_subnets
        self.n_input_nodes = n_input_nodes
        self.activation_func = activation_func
        self.n_hidden_layers = len(subnet_arch)

        all_biases = [] 
        all_weights = []
        n_hidden_nodes_prev = n_input_nodes
        for i, n_hidden_nodes in enumerate(subnet_arch + [1]):
            if i == 0:
                w = torch.nn.Parameter(torch.empty(size=(n_subnets, n_hidden_nodes_prev, n_hidden_nodes),
                                dtype=torch.float, requires_grad=True, device=device))
                b = torch.nn.Parameter(torch.empty(size=(n_subnets, n_hidden_nodes),
                                dtype=torch.float, requires_grad=True, device=device))
            elif i == self.n_hidden_layers:
                w = torch.nn.Parameter(torch.empty(size=(n_subnets, n_hidden_nodes_prev, 1),
                                dtype=torch.float, requires_grad=True, device=device))
                b = torch.nn.Parameter(torch.empty(size=(n_subnets, 1),
                                dtype=torch.float, requires_grad=True, device=device))
            else:
                w = torch.nn.Parameter(torch.empty(size=(n_subnets, n_hidden_nodes_prev, n_hidden_nodes),
                                dtype=torch.float, requires_grad=True, device=device))
                b = torch.nn.Parameter(torch.empty(size=(n_subnets, n_hidden_nodes),
                                dtype=torch.float, requires_grad=True, device=device))
            n_hidden_nodes_prev = n_hidden_nodes
            torch.nn.init.zeros_(b)
            for j in range(n_subnets):
                torch.nn.init.orthogonal_(w[j])
            all_biases.append(b)
            all_weights.append(w)
        self.all_biases = torch.nn.ParameterList(all_biases)
        self.all_weights = torch.nn.ParameterList(all_weights)

    def individual_forward(self, inputs, idx):

        xs = inputs
        for i in range(self.n_hidden_layers):
            xs = self.activation_func(torch.matmul(xs, self.all_weights[i][idx]) + self.all_biases[i][idx])
        outputs = torch.matmul(xs, self.all_weights[-1][idx]) + self.all_biases[-1][idx]
        return outputs

    def forward(self, inputs):

        xs = inputs
        for i in range(self.n_hidden_layers):
            xs = self.activation_func(torch.matmul(xs, self.all_weights[i])
                              + torch.reshape(self.all_biases[i], [self.n_subnets, 1, -1]))

        outputs = torch.matmul(xs, self.all_weights[-1]) + torch.reshape(self.all_biases[-1], [self.n_subnets, 1, -1])
        outputs = torch.squeeze(torch.transpose(outputs, 0, 1), dim=2)
        return outputs


class UnivariateOneHotEncodingLayer(torch.nn.Module):

    def __init__(self, num_classes_list, device):

        super(UnivariateOneHotEncodingLayer, self).__init__()

        self.class_bias = []
        self.global_bias = []
        self.num_classes_list = num_classes_list
        for i in range(len(num_classes_list)):
            cb = torch.nn.Parameter(torch.empty(size=(num_classes_list[i], 1),
                            dtype=torch.float, requires_grad=True, device=device))
            gb = torch.nn.Parameter(torch.empty(size=(1, 1),
                            dtype=torch.float, requires_grad=False, device=device))
            torch.nn.init.zeros_(cb)
            torch.nn.init.zeros_(gb)
            self.class_bias.append(cb)
            self.global_bias.append(gb)

    def forward(self, inputs, sample_weight=None, training=False):

        output = []
        for i in range(len(self.num_classes_list)):
            dummy = torch.nn.functional.one_hot(inputs[:, i].to(torch.int64),
                                    num_classes=self.num_classes_list[i]).to(torch.float)
            output.append(torch.matmul(dummy, self.class_bias[i]) + self.global_bias[i])
        output = torch.squeeze(torch.hstack(output))
        return output


class pyGAMNet(torch.nn.Module):

    def __init__(self, nfeature_index_list, cfeature_index_list, num_classes_list, subnet_arch, activation_func, device):

        super(pyGAMNet, self).__init__()

        self.nfeature_index_list = nfeature_index_list
        self.cfeature_index_list = cfeature_index_list
        self.num_classes_list = num_classes_list
        if len(self.nfeature_index_list) > 0:
            self.nsubnets = TensorLayer(len(nfeature_index_list), subnet_arch, 1, activation_func, device)
        if len(self.cfeature_index_list) > 0:
            self.csubnets = UnivariateOneHotEncodingLayer(num_classes_list, device)

    def forward(self, inputs):

        output = torch.zeros(size=(inputs.shape[0], inputs.shape[1]), dtype=torch.float)
        if len(self.nfeature_index_list) > 0:
            ntensor_inputs = torch.unsqueeze(torch.transpose(inputs[:, self.nfeature_index_list], 0, 1), 2)
            output[:, self.nfeature_index_list] = self.nsubnets(ntensor_inputs)
        if len(self.cfeature_index_list) > 0:
            ctensor_inputs = inputs[:, self.cfeature_index_list]
            output[:, self.cfeature_index_list] = self.csubnets(ctensor_inputs)
        return output
    
    
class pyInteractionNet(torch.nn.Module):

    def __init__(self, interaction_list, nfeature_index_list, cfeature_index_list, num_classes_list,
                 subnet_arch, activation_func=torch.nn.ReLU(), device="cpu"):
        super(pyInteractionNet, self).__init__()

        self.interaction_list = interaction_list
        self.n_interactions = len(interaction_list)
        self.nfeature_index_list = nfeature_index_list
        self.cfeature_index_list = cfeature_index_list
        self.num_classes_list = num_classes_list
        self.device = device

        self.n_inputs1 = []
        self.n_inputs2 = []
        for i in range(self.n_interactions):
            if self.interaction_list[i][0] in self.cfeature_index_list:
                self.n_inputs1.append(self.num_classes_list[self.cfeature_index_list.index(self.interaction_list[i][0])])
            else:
                self.n_inputs1.append(1)
                
            if self.interaction_list[i][1] in self.cfeature_index_list:
                self.n_inputs2.append(self.num_classes_list[self.cfeature_index_list.index(self.interaction_list[i][1])])
            else:
                self.n_inputs2.append(1)

        self.max_n_inputs = max([self.n_inputs1[i] + self.n_inputs2[i] for i in range(self.n_interactions)])
        self.subnets = TensorLayer(self.n_interactions, subnet_arch, self.max_n_inputs, activation_func, device)

    def preprocessing(self, inputs):

        preprocessed_inputs = []
        for i in range(self.n_interactions):
            interact_input_list = []
            idx1 = self.interaction_list[i][0]
            idx2 = self.interaction_list[i][1]
            if self.interaction_list[i][0] in self.cfeature_index_list:
                interact_input1 = torch.nn.functional.one_hot(inputs[:, idx1].to(torch.int64),
                                               num_classes=self.n_inputs1[i]).to(torch.float)
                interact_input_list.append(interact_input1)
            else:
                interact_input_list.append(inputs[:, [idx1]])
            if self.interaction_list[i][1] in self.cfeature_index_list:
                interact_input2 = torch.nn.functional.one_hot(inputs[:, idx2].to(torch.int64),
                                               num_classes=self.n_inputs2[i]).to(torch.float)
                interact_input_list.append(interact_input2)
            else:
                interact_input_list.append(inputs[:, [idx2]])

            if (self.n_inputs1[i] + self.n_inputs2[i]) < self.max_n_inputs:
                interact_input_list.append(torch.zeros(size=(inputs.shape[0], self.max_n_inputs - (self.n_inputs1[i] + self.n_inputs2[i])),
                                          dtype=torch.float, requires_grad=True, device=self.device))
            preprocessed_inputs.append(torch.hstack(interact_input_list))
        preprocessed_inputs = torch.hstack(preprocessed_inputs)
        return preprocessed_inputs

    def forward(self, inputs):

        tensor_inputs = torch.transpose(torch.reshape(self.preprocessing(inputs),
                                       [-1, self.n_interactions, self.max_n_inputs]), 0, 1)
        subnet_output = self.subnets(tensor_inputs)
        return subnet_output


class pyGAMINet(torch.nn.Module):

    def __init__(self, nfeature_index_list, cfeature_index_list, num_classes_list,
                 subnet_size_main_effect, subnet_size_interaction, activation_func,
                 heredity, mono_increasing_list, mono_decreasing_list, 
                 boundary_clip, normalize, min_value, max_value, mu_list, std_list, device):

        super(pyGAMINet, self).__init__()

        self.n_features = len(nfeature_index_list) + len(cfeature_index_list)
        self.nfeature_index_list = nfeature_index_list
        self.cfeature_index_list = cfeature_index_list
        self.num_classes_list = num_classes_list
        self.subnet_size_main_effect = subnet_size_main_effect
        self.subnet_size_interaction = subnet_size_interaction
        self.activation_func= activation_func
        self.heredity = heredity
        self.mono_increasing_list = mono_increasing_list
        self.mono_decreasing_list = mono_decreasing_list

        self.boundary_clip = boundary_clip
        self.normalize = normalize
        self.min_value = min_value
        self.max_value = max_value
        self.mu_list = mu_list
        self.std_list = std_list
        
        self.device = device
        self.interaction_status = False
        self.main_effect_blocks = pyGAMNet(nfeature_index_list=nfeature_index_list,
                                 cfeature_index_list=cfeature_index_list,
                                 num_classes_list=num_classes_list,
                                 subnet_arch=subnet_size_main_effect,
                                 activation_func=activation_func,
                                 device=device)
        self.main_effect_weights = torch.nn.Parameter(torch.empty(size=(self.n_features, 1),
                                dtype=torch.float, requires_grad=True, device=device))
        self.main_effect_switcher = torch.nn.Parameter(torch.empty(size=(self.n_features, 1),
                                dtype=torch.float, requires_grad=False, device=device))

        self.output_bias = torch.nn.Parameter(torch.empty(size=(1, ),
                                dtype=torch.float, requires_grad=True, device=device))
        torch.nn.init.zeros_(self.output_bias)
        torch.nn.init.ones_(self.main_effect_switcher)
        torch.nn.init.ones_(self.main_effect_weights)

    def init_interaction_blocks(self, interaction_list):

        if len(interaction_list) > 0:
            self.interaction_status = True
            self.n_interactions = len(interaction_list)
            self.interaction_blocks = pyInteractionNet(interaction_list=interaction_list,
                             nfeature_index_list=self.nfeature_index_list,
                             cfeature_index_list=self.cfeature_index_list,
                             num_classes_list=self.num_classes_list,
                             subnet_arch=self.subnet_size_interaction,
                             activation_func=self.activation_func,
                             device=self.device)
            self.interaction_weights = torch.nn.Parameter(torch.empty(size=(self.n_interactions, 1),
                                    dtype=torch.float, requires_grad=True, device=self.device))
            self.interaction_switcher = torch.nn.Parameter(torch.empty(size=(self.n_interactions, 1),
                                    dtype=torch.float, requires_grad=False, device=self.device))
            torch.nn.init.ones_(self.interaction_switcher)
            torch.nn.init.ones_(self.interaction_weights)

    def get_mono_loss(self, inputs, outputs=None, monotonicity=False):

        mono_loss = torch.tensor(0.0, requires_grad=True)
        if not monotonicity:
            return mono_loss

        grad = torch.autograd.grad(outputs=torch.sum(outputs),
                          inputs=inputs, create_graph=True)[0]
        if len(self.mono_increasing_list) > 0:
            mono_loss = mono_loss + torch.mean(torch.nn.ReLU()(-grad[:, self.mono_increasing_list]))
        if len(self.mono_decreasing_list) > 0:
            mono_loss = mono_loss + torch.mean(torch.nn.ReLU()(grad[:, self.mono_decreasing_list]))
        return mono_loss

    def get_clarity_loss(self, main_effect_outputs=None, interaction_outputs=None, sample_weight=None, clarity=False):

        clarity_loss = torch.tensor(0.0, requires_grad=True)
        if main_effect_outputs is None:
            return clarity_loss
        if interaction_outputs is None:
            return clarity_loss
        if not clarity:
            return clarity_loss

        for i, (k1, k2) in enumerate(self.interaction_blocks.interaction_list):
            if sample_weight is not None:
                clarity_loss = clarity_loss + torch.abs((main_effect_outputs[:, k1] 
                                           * interaction_outputs[:, i] * sample_weight.ravel()).mean())
                clarity_loss = clarity_loss + torch.abs((main_effect_outputs[:, k2] 
                                           * interaction_outputs[:, i] * sample_weight.ravel()).mean())
            else:
                clarity_loss = clarity_loss + torch.abs((main_effect_outputs[:, k1] 
                                           * interaction_outputs[:, i]).mean())
                clarity_loss = clarity_loss + torch.abs((main_effect_outputs[:, k2] 
                                           * interaction_outputs[:, i]).mean())
        return clarity_loss

    def forward(self, inputs, main_effect=True, interaction=True, clarity=False, monotonicity=False, sample_weight=None):

        main_effect_outputs = None
        interaction_outputs = None
        inputs.requires_grad = True
        outputs = self.output_bias * torch.ones(inputs.shape[0], 1)
        inputs = torch.max(torch.min(inputs, self.max_value), self.min_value) if self.boundary_clip else inputs
        inputs = (inputs - self.mu_list) / self.std_list if self.normalize else inputs
        if main_effect:
            main_effect_weights = self.main_effect_switcher * self.main_effect_weights
            main_effect_outputs = self.main_effect_blocks(inputs) * main_effect_weights.ravel()
            outputs = outputs + main_effect_outputs.sum(1, keepdim=True)
        if interaction and self.interaction_status:
            interaction_weights = self.interaction_switcher * self.interaction_weights
            interaction_outputs = self.interaction_blocks(inputs) * interaction_weights.ravel()
            outputs = outputs + interaction_outputs.sum(1, keepdim=True)

        self.mono_loss = self.get_mono_loss(inputs, outputs, monotonicity)
        self.clarity_loss = self.get_clarity_loss(main_effect_outputs, interaction_outputs, sample_weight, clarity)
        return outputs
