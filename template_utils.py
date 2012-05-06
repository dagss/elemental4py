#
# Python code for use in the templates
#

def as_distribution_list(dists):
    """
    Turn a string "MC,MR MC,STAR" into [('MC', 'MR'), ('MC', 'STAR')],
    if necesarry.
    """
    if isinstance(dists, str):
        return [tuple(x.split(',')) for x in dists.split()]
    else:
        return dists
