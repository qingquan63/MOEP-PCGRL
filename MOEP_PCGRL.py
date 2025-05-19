import argparse
from SDEMOEA_ext import SDE_parent_selection, SDE_env_selection
from src.smb.asyncsimlt import AsycSimltPool
from src.drl.train_async import *
from src.gan.adversarial_train import *
from src.drl.train_sinproc import set_SAC_parser, train_SAC
from src.drl.egsac.train_egsac import set_EGSAC_parser, train_EGSAC
from src.drl.sunrise.train_sunrise import train_SUNRISE, set_SUNRISE_args
from src.drl.dvd import set_DvDSAC_parser, train_DvDSAC
import copy
from torch import tensor
from pymoo.operators.crossover.sbx import cross_sbx
from pymoo.core.variable import Real, get
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import  Mutation
from pymoo.operators.mutation.pm import mut_pm
import pathlib
import os
from itertools import chain
from src.utils.datastruct import RingQueue
from pymoo.decomposition.pbi import PBI
import matplotlib.pyplot as plt
nz = 20



class GenPolicy:
    def __init__(self, n=5):
        self.n = n # Number of segments in an observation

    @abstractmethod
    def step(self, obs):
        pass

    @staticmethod
    @abstractmethod
    def from_path(path, **kwargs):
        pass

    def reset(self):
        pass


class RLGenPolicy(GenPolicy):
    def __init__(self, model, n, device='cuda:0'):
        self.model = model
        super(RLGenPolicy, self).__init__(n)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.meregs = []

    def step(self, obs):
        obs = process_obs(obs, device=self.device)
        b, d = obs.shape
        if d < nz * self.n:
            obs = torch.cat([torch.zeros([b, nz * self.n - d], device=self.device), obs], dim=-1)
        with torch.no_grad():
            # mus, sigmas, betas = self.model.get_intermediate(obs)
            # print(mus[0].cpu().numpy(), '\n', betas[0].cpu().numpy(), '\n')
            model_output, _ = self.model(obs)
        return torch.clamp(model_output, -1, 1).squeeze().cpu().numpy()

    @staticmethod
    def from_path(path, device='cuda:0'):
        model = torch.load(getpath(f'{path}'), map_location=device)
        # n = load_cfgs(path, 'N')
        n = 5
        return RLGenPolicy(model, n, device)


def evaluate_play_div(lvls, rfunc='default', dest_path='', parallel=1, eval_pool=None):
    # src/smb/asyncsimlt.py
    # src/env/rfunc.py
    internal_pool = eval_pool is None
    if internal_pool:
        eval_pool = AsycSimltPool(parallel, rfunc_name=rfunc, verbose=False, test=True)
    res = []
    resss = []
    playability = []
    for lvl in lvls:
        res_ = []
        playability_ = []
        eval_pool.put('evaluate', (0, str(lvl)))
        buffer = eval_pool.get()
        for _, item in buffer:
            for r in zip(*item.values()):
                # print(r)
                res_.append(r[0]+r[1])
                playability_.append(r[2])
            res.append(res_)
            playability.append(playability_)
            resss.append([sum(r) for r in zip(*item.values())])
    if internal_pool:
        buffer = eval_pool.close()
    else:
        buffer = eval_pool.get(True)
    for _, item in buffer:
        res_ = []
        playability_ = []
        resss.append([sum(r) for r in zip(*item.values())])
        for r in zip(*item.values()):
            res_.append(r[0]+r[1])
            playability_.append(r[2])
        res.append(res_)
        playability.append(playability_)
    if len(dest_path):
        np.save(dest_path, res)
    div_val  = [sum(item) for item in res]
    play_val  = [sum(item) for item in playability]
    concat_val = [sum(item) for item in resss]
    return 0-np.mean(play_val), np.mean(div_val), np.mean(concat_val)


def evaluate_mpd(lvls, parallel=2):
    # src/smb/asyncsimlt.py
    task_datas = [[] for _ in range(parallel)]
    for i, (A, B) in enumerate(combinations(lvls, 2)):
        # lvlA, lvlB = lvls[i * 2], lvls[i * 2 + 1]
        task_datas[i % parallel].append((str(A), str(B)))

    hms, dtws = [], []
    eval_pool = AsycSimltPool(parallel, verbose=False)
    for task_data in task_datas:
        eval_pool.put('mpd', task_data)
    res = eval_pool.get(wait=True)
    for task_hms, _ in res:
        hms += task_hms
    return np.mean(hms)


def load_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_gan = subparsers.add_parser('gan', help='Train GAN')
    set_GAN_parser(parser_gan)
    parser_gan.set_defaults(entry=train_GAN)

    parser_sac = subparsers.add_parser('sac', help='Train SAC')
    set_SAC_parser(parser_sac)
    parser_sac.set_defaults(entry=train_SAC)

    parser_asyncsac = subparsers.add_parser('asyncsac', help='Train Asynchronous SAC')
    set_AsyncSAC_parser(parser_asyncsac)
    parser_asyncsac.set_defaults(entry=train_AsyncSAC)

    parser_egsac = subparsers.add_parser('egsac', help='Train Episodic Generative SAC')
    set_EGSAC_parser(parser_egsac)
    parser_egsac.set_defaults(entry=train_EGSAC)

    parser_ncesac = subparsers.add_parser('ncesac', help='Train Negatively Correlated Ensemble SAC')
    set_NCESAC_parser(parser_ncesac)
    parser_ncesac.set_defaults(entry=train_NCESAC)

    parser_pmoesac = subparsers.add_parser('pmoe', help='Train PMOE')
    set_PMOESAC_parser(parser_pmoesac)
    parser_pmoesac.set_defaults(entry=train_PMOESAC)

    parser_sunrise = subparsers.add_parser('sunrise', help='Train SUNRISE')
    set_SUNRISE_args(parser_sunrise)
    parser_sunrise.set_defaults(entry=train_SUNRISE)

    parser_dvd = subparsers.add_parser('dvd', help='Train DvD')
    set_DvDSAC_parser(parser_dvd)
    parser_dvd.set_defaults(entry=train_DvDSAC)

    args = parser.parse_args()
    return args


def load_agents(args, actor_path, device):
    obs_dim = 100
    act_dim = 20
    agent = get_NCESAC(args, '/home/zqq/PycharmProjects/NCERL-Diverse-PCG/training_data/NCESAC0', device, obs_dim, act_dim)
    if isinstance(actor_path, str):
        actor_ = torch.load(actor_path)
        agent.actor.net.load_state_dict(actor_.state_dict())
        return agent
    else:
        agents = []
        for pa in actor_path:
            actor_ = torch.load(pa)
            agent.actor.net.load_state_dict(actor_.state_dict())
            agents.append(copy.deepcopy(agent))
        return agents


def indiv2sel_weights(p):
    thred = 0.2
    pp = copy.deepcopy(p) 
    sel_index = np.where(pp >= thred)[0]
    pp[pp < thred] = 0
    comp_cost = np.sum(pp > 0)
    sel_weights = np.around(pp, decimals=5, out=None)
    sel_weights = sel_weights / np.sum(sel_weights)
    sel_weights = np.around(sel_weights, decimals=5, out=None)
    sel_weights[sel_index[0]] = 1 - np.sum(sel_weights[sel_index[1:]])
    
    return sel_weights, comp_cost


def make_decision_MM_new(agent, obs, device):
    agent.actor.net.requires_grad_(False)
    muss, stdss, betas = agent.actor.net.get_intermediate(torch.tensor(obs).to(device))
    subpolicies = Independent(Normal(muss, stdss), 1)

    actionss = torch.clamp(subpolicies.rsample(), -1, 1)
    return actionss


def next_prunned_seg(basemodels, invid, obs_buffer, device, GMM_size):
    obs = torch.tensor(np.concatenate(obs_buffer.to_list(), axis=-1), dtype=torch.float).to(device)
    total_segss = torch.zeros((len(basemodels), obs.shape[0], GMM_size, 20)).to(device)
    next_segss = torch.zeros((obs.shape[0], 20)).to(device)
    
    count = 0
    for i, basemodel in enumerate(basemodels):
        gen_segs = make_decision_MM_new(basemodel, obs, device)
        total_segss[i, :, :, :] = gen_segs
    
    for b in range(len(basemodels)):
        for g in range(GMM_size):
            next_segss += invid[count] * total_segss[b, :, g, :]
            count += 1
    next_segss = torch.clamp(next_segss, -1, 1)
    return next_segss.detach().cpu().numpy()


def evaluation_indiv(pop, basemodels, eval_data_, device, level_iter, decoder, GMM_size, eval_levels_):
    metrics = []
    pop_levels = []
    pop_level_latvecs = []
    parallel = 50
    for i, p in enumerate(pop):
        # print(f'individual {i}: data', end="")
        print(f'   individual {i}')
        eval_data = np.vstack(eval_data_)
        # eval_levels = copy.deepcopy(eval_levels_)
        sel_weights, comp_cost = indiv2sel_weights(p)
        levels = []
        level_latvecs = []
        eval_levels = []
        n = 5
        
        obs_buffer = RingQueue(n)
        for _ in range(level_iter): obs_buffer.push(np.zeros_like(eval_data))
        obs_buffer.push(eval_data)
        
        for d in range(len(eval_data_)):
            eval_levels.append([process_onehot(decoder(tensor(eval_data_[d]).to(device).view(-1, nz, 1, 1)))])
        
        for it in range(level_iter):
            next_seg = next_prunned_seg(basemodels, sel_weights, obs_buffer, device, GMM_size)
            obs_buffer.push(next_seg)
            z = torch.tensor(next_seg, device=device).view(-1, nz, 1, 1)
            next_lvls = process_onehot(decoder(z))
            
            for i, nl in enumerate(next_lvls):
                eval_levels[i].append(nl)
            
        batchs = [[lvlhcat(item) for item in eval_levels] for _ in range(1)]
        res = list(chain(*batchs))
        playability, lvl_div, concat_div = evaluate_play_div(res, parallel=parallel)
        div_val = evaluate_mpd(res, parallel=parallel)
        metrics.append([concat_div, div_val, comp_cost, playability, lvl_div])    
        pop_levels.append(res)
        pop_level_latvecs.append(res)
        
    metrics = np.vstack(metrics)
    
    return metrics, pop_levels, pop_level_latvecs
    

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def touranment_selection(fitness, num_parents):
    parents1 = []
    for i in range(int(num_parents/2)):
        idx = np.random.permutation(fitness.shape[0])[0:2]
        if fitness[idx[0]] > fitness[idx[1]]:
            parents1.append(idx[0])
        else:
            parents1.append(idx[1])
    parents2 = []
    for i in range(int(num_parents/2)):
        idx = np.random.permutation(fitness.shape[0])[0:2]
        if fitness[idx[0]] > fitness[idx[1]]:
            parents2.append(idx[0])
        else:
            parents2.append(idx[1])
    return np.hstack((np.array(parents1).reshape(-1, 1), np.array(parents2).reshape(-1, 1)))


class SimulatedBinaryCrossover(Crossover):

    def __init__(self,
                 prob_var=0.5,
                 eta=15,
                 prob_exch=1.0,
                 prob_bin=0.5,
                 n_offsprings=2,
                 xl=-1,
                 xu=1,
                 **kwargs):
        super().__init__(2, n_offsprings, **kwargs)

        self.prob_var = Real(prob_var, bounds=(0.1, 0.9))
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, None))
        self.prob_exch = Real(prob_exch, bounds=(0.0, 1.0), strict=(0.0, 1.0))
        self.prob_bin = Real(prob_bin, bounds=(0.0, 1.0), strict=(0.0, 1.0))
        self.xl = xl
        self.xu = xu

    def do(self, X):
        _, n_matings, _ = X.shape

        # get the parameters required by SBX
        eta, prob_var, prob_exch, prob_bin = get(self.eta, self.prob_var, self.prob_exch, self.prob_bin,
                                                 size=(n_matings, 1))

        # set the binomial probability to zero if no exchange between individuals shall happen
        rand = np.random.random((len(prob_bin), 1))
        prob_bin[rand > prob_exch] = 0.0

        Q = cross_sbx(X.astype(float), self.xl, self.xu, eta, prob_var, prob_bin)

        if self.n_offsprings == 1:
            rand = np.random.random(size=n_matings) < 0.5
            Q[0, rand] = Q[1, rand]
            Q = Q[[0]]

        return Q


class PolynomialMutation(Mutation):

    def __init__(self, prob=0.9, eta=20, at_least_once=False, xl=0, xu=0, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, 100.0))
        self.xl = xl
        self.xu = xu

    def do(self, X, prob_var, params=None, **kwargs):
        X = X.astype(float)

        eta = get(self.eta, size=len(X))
        
        prob_var = get( min(0.5, prob_var), size=len(X))
        Xp = mut_pm(X, self.xl, self.xu, eta, prob_var, at_least_once=self.at_least_once)

        return Xp


def convert_to_objs(metrics_, preference, theta):
    metrics = copy.deepcopy(np.array(metrics_))
    metrics[metrics[:, 0] < 0, 0] = lb[0]
    
    
    lvl_div = copy.deepcopy(metrics[:, 0])
    gp_div = copy.deepcopy(metrics[:, 1])
    lvl_div = (lvl_div - lb[0])/(ub[0]-lb[0])
    gp_div = (gp_div - lb[1])/(ub[1]-lb[1])
    divs = np.hstack((lvl_div.reshape(-1, 1), gp_div.reshape(-1, 1)))

    # divs: larger is better
    # div_obj: smaller is better
    pbi = PBI(theta=theta)
    fitness = pbi(0 - divs, weights=preference)
    div_obj = fitness+10
    
    
    objs = np.zeros((metrics.shape[0], 2))
    objs[:, 0] = div_obj
    objs[:, 1] = (metrics[:, 2]-lb[2])/(ub[2]-lb[2])
    return objs


def to_imgs(lvl, path):
    for i, p_lvl in enumerate(lvl):
        p_lvl.to_img(f"{path}-{i}.png")


if __name__ == '__main__':
    
    args = load_args()
    if args.prunning_plan in [1, 2, 3]: 
        seed = args.randseed
        sel_basemode = [2, 29]
        
        
    elif args.prunning_plan in [4, 5]:
        seed = args.randseed
        sel_basemode = [27, 29]
    
    elif args.prunning_plan in [6, 7]:
        seed = args.randseed
        sel_basemode = [2, 8]
    
    elif args.prunning_plan in [8, 9, 10]:
        seed = args.randseed
        sel_basemode = [12, 20]
        
    elif args.prunning_plan in [11, 12]:
        seed = args.randseed
        sel_basemode = [13]
    
    start_time = time.time()
    setup_seed(seed)
    pathlib.Path(f'/home/zqq/PycharmProjects/NCERL-Diverse-PCG/prunning/plan-{args.prunning_plan}/{seed}/').mkdir(parents=True, exist_ok=True)
    popsize = 4
    num_eval_data = 20
    theta = 2
    basemodel_infos = [
                       [1, 'test_data/varpm-lgp/l0.5_m5/t1/policy.pth', 42.9999007265229, 800.9761683366734],
                       [2, 'test_data/varpm-lgp/l0.5_m5/t2/policy.pth',26.699051739556605,1461.5931543086172],
                       [3, 'test_data/varpm-lgp/l0.5_m5/t3/policy.pth',30.888331870826136, 1276.1629338677355],
                       [4, 'test_data/varpm-lgp/l0.5_m5/t4/policy.pth',38.55359198232257,1008.8921683366733],
                       [5, 'test_data/varpm-lgp/l0.5_m5/t5/policy.pth',36.01475478975167,1147.6106773547094],
                       
                       [6, 'test_data/varpm-lgp/l0.4_m5/t1/policy.pth',41.24102936228133,965.9122805611222],
                       [7, 'test_data/varpm-lgp/l0.4_m5/t2/policy.pth',42.038790271617586,825.3660440881764],
                       [8, 'test_data/varpm-lgp/l0.4_m5/t3/policy.pth', 39.20085595530382,1025.4582364729458],
                       [9, 'test_data/varpm-lgp/l0.4_m5/t4/policy.pth',46.45056951867469,499.91285771543085],
                       [10, 'test_data/varpm-lgp/l0.4_m5/t5/policy.pth',32.780583831779744,1159.4189579158317],
                       
                       [11, 'test_data/varpm-lgp/l0.3_m5/t1/policy.pth',31.7883730705128,1234.4345090180361],
                       [12, 'test_data/varpm-lgp/l0.3_m5/t2/policy.pth',40.30886284369352,902.3904048096192],
                       [13, 'test_data/varpm-lgp/l0.3_m5/t3/policy.pth',41.445857270413974,855.0215150300601],
                       [14, 'test_data/varpm-lgp/l0.3_m5/t4/policy.pth',33.785467390435784,1182.3381322645291],
                       [15, 'test_data/varpm-lgp/l0.3_m5/t5/policy.pth',39.226128093208246,965.9540601202405],
                       
                       [16, 'test_data/varpm-lgp/l0.2_m5/t1/policy.pth',45.34054872618817,704.7894268537074],
                       [17, 'test_data/varpm-lgp/l0.2_m5/t2/policy.pth',46.02509618169058,493.10343887775554],
                       [18, 'test_data/varpm-lgp/l0.2_m5/t3/policy.pth',46.19775768516803,535.2210901803608],
                       [19, 'test_data/varpm-lgp/l0.2_m5/t4/policy.pth',43.01623732574747,727.2309659318637],
                       [20, 'test_data/varpm-lgp/l0.2_m5/t5/policy.pth',45.95639391979561,625.2883366733467],
                       
                       [21, 'test_data/varpm-lgp/l0.1_m5/t1/policy.pth',45.77849789300454,559.9656513026052],
                       [22, 'test_data/varpm-lgp/l0.1_m5/t2/policy.pth',46.635948757257076,503.3083126252505],
                       [23, 'test_data/varpm-lgp/l0.1_m5/t3/policy.pth',45.66268973933881,565.6079278557114],
                       [24, 'test_data/varpm-lgp/l0.1_m5/t4/policy.pth',46.09823993506838,437.5692264529058],
                       [25, 'test_data/varpm-lgp/l0.1_m5/t5/policy.pth',46.66952100146137,381.0810901803607],
                       
                       [26, 'test_data/varpm-lgp/l0.0_m5/t1/policy.pth',46.4805876431463,430.6837354709419],
                       [27, 'test_data/varpm-lgp/l0.0_m5/t2/policy.pth',46.10226448879111,532.9708857715431],
                       [28, 'test_data/varpm-lgp/l0.0_m5/t3/policy.pth',46.53720763373379,316.4156472945892],
                       [29, 'test_data/varpm-lgp/l0.0_m5/t4/policy.pth',47.15365110339178,246.37194388777556],
                       [30, 'test_data/varpm-lgp/l0.0_m5/t5/policy.pth',46.06446848423699,495.94703006012026],
                       ]
    basemodel_paths = []
    basemodel_perf = []
    for sel in sel_basemode:
        basemodel_paths.append(basemodel_infos[sel-1][1])
        basemodel_perf.append(basemodel_infos[sel-1][2:])
    
    GMM_size = 5
    num_basemodels = len(basemodel_paths) * GMM_size
    eta = 20
    prob_var = 0.9
    prob_bin = 0.5
    
    Total_gen = 21
    
    nd = num_basemodels
    level_iter = 7   # TODO
    
    init_set = np.load(getpath('analysis/initial_seg.npy'))
    
    # reward, diversity, comp_cost
    ub = np.array([50, 1500, num_basemodels])
    lb = np.array([0, 0, 0])
    ty = 1
    
    basemodel_perf = np.vstack(basemodel_perf)
    basemodel_perf_norm = copy.deepcopy(basemodel_perf)
    basemodel_perf_norm[:, 0] = (basemodel_perf[:, 0]-lb[0])/(ub[0]-lb[0])
    basemodel_perf_norm[:, 1] = (basemodel_perf[:, 1]-lb[1])/(ub[1]-lb[1])

    if args.prunning_plan in [2, 5, 6, 9]:
        real_preference = np.mean(basemodel_perf, axis=0)
    elif args.prunning_plan == 1:
        real_preference = np.mean(basemodel_perf, axis=0)
        real_preference[0] = real_preference[0] - 13
    elif args.prunning_plan == 3:
        real_preference = np.mean(basemodel_perf, axis=0)
        real_preference[0] = real_preference[0] + 18
    elif args.prunning_plan == 4:
        real_preference = np.mean(basemodel_perf, axis=0)
        real_preference[0] = real_preference[0] - 38
    elif args.prunning_plan == 7:
        real_preference = np.mean(basemodel_perf, axis=0)
        real_preference[0] = real_preference[0] + 60
    elif args.prunning_plan == 8:
        real_preference = np.mean(basemodel_perf, axis=0)
        real_preference[0] = real_preference[0] - 25
    elif args.prunning_plan == 10:
        real_preference = np.mean(basemodel_perf, axis=0)
        real_preference[0] = real_preference[0] + 60
    elif args.prunning_plan == 11:
        real_preference = np.mean(basemodel_perf, axis=0)
        real_preference[0] = real_preference[0] - 10
    elif args.prunning_plan == 12:
        real_preference = np.mean(basemodel_perf, axis=0)
        real_preference[0] = real_preference[0] + 10
        
        
    real_preference = 3 * real_preference
    
    real_preference_norm = np.zeros(2)
    real_preference_norm[0] = (real_preference[0]-lb[0])/(ub[0]-lb[0])
    real_preference_norm[1] = (real_preference[1]-lb[1])/(ub[1]-lb[1])
    
    real_preference_norm = real_preference_norm * 2
    
    xu = np.ones(nd)
    xl = np.zeros(nd)
    
    # load basemodels
    actor_path = 'training_data/NCESAC0/policy1000000.pth'
    device = 'cpu' if args.gpuid < 0 or not torch.cuda.is_available() else f'cuda:{args.gpuid}'
    agents = load_agents(args, basemodel_paths, device)
    
    # parpare evaluation data
    decoder = get_decoder('models/decoder.pth', 'cuda:0')
    evalpool = AsycSimltPool(args.n_workers, args.queuesize, args.rfunc, verbose=False)
    rfunc = importlib.import_module('src.env.rfuncs').__getattribute__(f'{args.rfunc}')()
    env = AsyncOlGenEnv(rfunc.get_n(), get_decoder('models/decoder.pth'), evalpool, args.eplen, device=device)
    eval_data = []
    eval_levels = []
    total_sel_lvls = []
    
    for i in range(num_eval_data):
        sel_lvls = np.random.permutation(len(env.initvec_set))[0]
        st_seg = env.initvec_set[sel_lvls]
        eval_levels.append(process_onehot(decoder(tensor(st_seg).to(device).view(-1, nz, 1, 1))))
        eval_data.append(st_seg)
        total_sel_lvls.append(sel_lvls)
    
    init_test_vec = np.load(getpath('analysis/initial_seg.npy'))
    test_data = []
    test_levels = []
    for i in range(init_test_vec.shape[0]):
        sel_lvls = np.random.permutation(len(env.initvec_set))[0]
        st_seg = env.initvec_set[sel_lvls]
        test_levels.append(process_onehot(decoder(tensor(st_seg).to(device).view(-1, nz, 1, 1))))
        test_data.append(st_seg)
        
    st1 = time.time()
    
    # initialisation
    pop = []
    for i in range(popsize):
        p = np.random.uniform(0, 1, nd)
        pop.append(p)
    pop = np.array(pop)
    
    test_metrics = []
    test_levels = []
    test_level_latvecs = []
    
    
    # evaluation
    pop_metrics, pop_levels, pop_level_latvecs = evaluation_indiv(pop, agents, eval_data, device, level_iter, decoder, GMM_size, eval_levels)
    pop_objs = convert_to_objs(pop_metrics, real_preference_norm, theta)
    
    
    
    total_time = [st1-start_time]
    # test data evaluation
    for gen in range(1, Total_gen):
        st1 = time.time()
        print(f'Generation {gen}')
        
        # touranment selection
        fitness = SDE_parent_selection(pop_objs)
        mating_pool = touranment_selection(fitness, popsize)
        
        # reproduction
        mp = np.zeros((2, mating_pool.shape[0], pop.shape[1]))
        mp[0, :, :] = pop[mating_pool[:, 0], :]
        mp[1, :, :] = pop[mating_pool[:, 1], :]
        sbc = SimulatedBinaryCrossover(prob_var=prob_var, eta=10, prob_exch=1.0, prob_bin=prob_bin, xl=xl, xu=xu)
        off = sbc.do(mp)
        off = np.vstack((off[0, :, :], off[1, :, :]))
        pm = PolynomialMutation(prob=0.9, eta=20, at_least_once=False, xl=xl, xu=xu)
        off = pm.do(off, prob_var=5/num_basemodels)
        
        # offspring model evaluation
        off_metrics, off_levels, off_level_latvecs = evaluation_indiv(off, agents, eval_data, device, level_iter, decoder, GMM_size, eval_levels)
        off_objs = convert_to_objs(off_metrics, real_preference_norm, theta)
        
        # environmental selection
        pop_off = np.vstack((pop, off))
        pop_off_objs = np.vstack((pop_objs, off_objs))
        pop_off_metrics = np.vstack((pop_metrics, off_metrics))
        pop_off_levels = pop_levels + off_levels
        pop_off_level_latvecs = pop_level_latvecs + off_level_latvecs
    
        sel_idx = SDE_env_selection(pop_off_objs, popsize)
        pop = pop_off[sel_idx, :]
        pop_objs = pop_off_objs[sel_idx, :]
        pop_metrics = pop_off_metrics[sel_idx, :]
        pop_levels = [pop_off_levels[int(i)] for i in sel_idx]
        pop_level_latvecs = [pop_off_level_latvecs[int(i)] for i in sel_idx]
        
        ed_time = time.time()
        total_time.append(ed_time-st1)
        pathlib.Path(f'/home/zqq/PycharmProjects/NCERL-Diverse-PCG/prunning/plan-{args.prunning_plan}/{seed}/gen{gen}').mkdir(parents=True, exist_ok=True)
        np.savetxt(f"/home/zqq/PycharmProjects/NCERL-Diverse-PCG/prunning/plan-{args.prunning_plan}/{seed}/pop-{gen}.txt", pop)
        np.savetxt(f"/home/zqq/PycharmProjects/NCERL-Diverse-PCG/prunning/plan-{args.prunning_plan}/{seed}/pop_objs-{gen}.txt", pop_objs)
        np.savetxt(f"/home/zqq/PycharmProjects/NCERL-Diverse-PCG/prunning/plan-{args.prunning_plan}/{seed}/pop_metrics-{gen}.txt", pop_metrics)
        np.savetxt(f"/home/zqq/PycharmProjects/NCERL-Diverse-PCG/prunning/plan-{args.prunning_plan}/{seed}/timecost-{gen}.txt", np.array(total_time))
        
        for i in range(popsize):
            to_imgs(pop_levels[i], f"/home/zqq/PycharmProjects/NCERL-Diverse-PCG/prunning/plan-{args.prunning_plan}/{seed}/gen{gen}/pop_gen{gen}_p{i}")
        
        
    # test_data = eval_data
        if (np.mod(gen, 20) == 0) and (gen > 0):
            print(f"test {gen}")
            test_metrics = []
            test_levels = []
            test_level_latvecs = []
            test_metric, test_level, test_level_latvec = evaluation_indiv(pop, agents, test_data, device, level_iter, decoder, GMM_size, None)    
            np.savetxt(f"/home/zqq/PycharmProjects/NCERL-Diverse-PCG/prunning/plan-{args.prunning_plan}/{seed}/test_metrics-{gen}.txt", test_metric)
            
            
            plt.figure(figsize=(12, 5))
            plt.subplot(1,2,1)
            for i in range(len(basemodel_perf)):
                plt.annotate(f'({basemodel_perf[i,0]:.2f}, {basemodel_perf[i,1]:.2f})', 
                            (basemodel_perf[i,0]-5, basemodel_perf[i,1]),
                            xytext=(5, 5), textcoords='offset points')
            plt.scatter(basemodel_perf[:, 0], basemodel_perf[:, 1], c='red')
            plt.scatter(test_metric[:, 0], test_metric[:, 1], c='green')
            plt.plot([0, real_preference[0]], [0, real_preference[1]])
            plt.xlim(0, 50)
            plt.ylim(0, 1500)
            plt.xlabel("Reward")
            plt.ylabel("Diversity")
            plt.title(f"Generation {gen}")
            
            plt.subplot(1,2,2)
            test_pop = np.random.rand(50000, 2)
            pbi = PBI(theta=theta)
            div_obj = pbi(0-test_pop, weights=real_preference_norm)
            plt.scatter(test_pop[:, 0], test_pop[:, 1], c=div_obj, cmap='viridis')
            for i in range(len(basemodel_perf)):
                plt.annotate(f'({basemodel_perf_norm[i,0]:.2f}, {basemodel_perf_norm[i,1]:.2f})', 
                            (basemodel_perf_norm[i,0]-0.15, basemodel_perf_norm[i,1]),
                            xytext=(5, 5), textcoords='offset points')
            plt.scatter(basemodel_perf_norm[:, 0], basemodel_perf_norm[:, 1], c='red')    
            plt.scatter(real_preference_norm[0], real_preference_norm[1])
            plt.plot([0, real_preference_norm[0]], [0, real_preference_norm[1]])
            lvl_obj = (test_metric[:, 0]-lb[0])/(ub[0]-lb[0])
            gp_obj = (test_metric[:, 1]-lb[1])/(ub[1]-lb[1])
            plt.scatter(lvl_obj, gp_obj, c='green')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("Reward")
            plt.ylabel("Diversity")
            plt.title(f"Generation {gen}")
            plt.show()
            plt.savefig(f"prunning/plan-{args.prunning_plan}/{seed}/gen{gen}-test.png")
            plt.close()

        
    print("OK")


    
