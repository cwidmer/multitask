import expenv
import shogun_factory_new

tax = expenv.Taxonomy.get(18070).data
shogun_tax = shogun_factory_new.create_taxonomy(tax)

print shogun_tax.get_task_similarity("toy_0", "toy_7")
print shogun_tax.get_task_similarity("toy_0", "toy_1")
print shogun_tax.get_task_similarity("toy_0", "toy_0")

print shogun_tax.get_task_similarity("inner_0", "inner_1")
print shogun_tax.get_task_similarity("inner_0", "inner_3")
print shogun_tax.get_task_similarity("inner_1", "inner_3")

print shogun_tax.get_task_similarity("toy_0", "root")

