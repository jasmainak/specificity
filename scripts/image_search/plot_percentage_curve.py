from scipy.io import loadmat
import matplotlib.pyplot as plt

mat = loadmat('../../data/search_results/percentage_results.mat',
              squeeze_me=True, struct_as_record=False)

margin = range(1, 11)
kwargs = dict(linewidth=2, markersize=8, markeredgewidth=2,
              markerfacecolor='w')
plt.figure(figsize=(7, 6))

plt.plot(margin, mat['pascal'].stats_gt.y[:10, 0], 'b-o', markeredgecolor='b',
         label='GT-Spec (PASCAL-50S)', **kwargs)
plt.plot(margin, mat['clipart'].stats_gt.y[:10, 0], 'r-s', markeredgecolor='r',
         label='GT-Spec (ABSTRACT-50S)', **kwargs)
plt.plot(margin, mat['pascal'].stats_s.y[:10, 0], 'b--o', markeredgecolor='b',
         label='P-Spec (PASCAL-50S)', **kwargs)
plt.plot(margin, mat['clipart'].stats_s.y[:10, 0], 'r--s', markeredgecolor='r',
         label='P-Spec (ABSTRACT-50S)', **kwargs)

plt.yticks(range(0, 50, 5))

# remove top and right axis
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.legend(numpoints=1)
plt.xlim((0, 10.15))
plt.ylim((-0.5, 47))
plt.xlabel('margin K by which baseline is beaten', fontsize=16)
plt.ylabel('% queries baseline is beaten by at least K', fontsize=16)
plt.rc("xtick", direction="out")
plt.rc("ytick", direction="out")
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

# TODO: use Helvetica fonts :)

print('Rank of baseline for pascal is %0.2f' % mat['pascal'].stats_b.rank)
print('Rank of P-Spec for pascal is %0.2f' % mat['pascal'].stats_s.rank)
print('Rank of GT-Spec for pascal is %0.2f' % mat['pascal'].stats_gt.rank)
print('Rank of baseline for clipart is %0.2f' % mat['clipart'].stats_b.rank)
print('Rank of P-Spec for clipart is %0.2f' % mat['clipart'].stats_s.rank)
print('Rank of GT-Spec for clipart is %0.2f' % mat['clipart'].stats_gt.rank)

plt.tight_layout(pad=0.1)

plt.show()

# save figure
fig = plt.gcf()
fig.savefig('../../plots/percentage_results.pdf', bbox='tight',
            pad_inches=0)
