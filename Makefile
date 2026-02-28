# Makefile for compressor_2 Evolution

KCONFIG := Kconfig
CONFIG := .config
DEFCONFIG := defconfig

.PHONY: help menuconfig defconfig savedefconfig evolve clean install_requirements

help:
	@echo "compressor_2 Evolution - Available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install_requirements  - Install Python deps (pip install -r requirements.txt)"
	@echo ""
	@echo "  Configuration:"
	@echo "    menuconfig     - Interactive configuration menu (ncurses)"
	@echo "    defconfig      - Load default configuration"
	@echo "    savedefconfig  - Save current config as defconfig"
	@echo ""
	@echo "  Execution:"
	@echo "    evolve         - Run the evolution loop"
	@echo ""
	@echo "  Cleanup:"
	@echo "    clean          - Remove evolution outputs (keeps config)"
	@echo ""

install_requirements:
	pip install -r requirements.txt

menuconfig:
	@python3 -c "from kconfiglib import Kconfig; import menuconfig; \
		import os; os.environ['KCONFIG_CONFIG'] = '$(CONFIG)'; \
		menuconfig.menuconfig(Kconfig('$(KCONFIG)'))"

defconfig:
	@if [ -f $(DEFCONFIG) ]; then \
		cp $(DEFCONFIG) $(CONFIG); \
		echo "Loaded default configuration from $(DEFCONFIG)"; \
	else \
		echo "Error: $(DEFCONFIG) not found"; \
		exit 1; \
	fi

savedefconfig:
	@if [ -f $(CONFIG) ]; then \
		cp $(CONFIG) $(DEFCONFIG); \
		echo "Saved current configuration to $(DEFCONFIG)"; \
	else \
		echo "Error: $(CONFIG) not found"; \
		exit 1; \
	fi

evolve:
	@if [ ! -f $(CONFIG) ]; then \
		echo "No $(CONFIG) found. Run 'make menuconfig' or 'make defconfig' first."; \
		exit 1; \
	fi
	cd .. && python3 -m compressor_2 evolve --config compressor_2/$(CONFIG)

clean:
	rm -f outputs/evolution/deltas_current.json
	rm -f outputs/evolution/deltas_best.json
	rm -f outputs/evolution/history.json
	rm -f outputs/evolution/cot_summary.txt
	rm -f outputs/evolution/processor.py
	rm -f outputs/evolution/processor_best.py
	rm -f outputs/evolution/evolution_lengths.png
	rm -f outputs/evolution/ga_fitness.png
	rm -f outputs/evolution/ga_objective.png
	rm -f outputs/evolution/evolution_state.json
	rm -f outputs/evolution/ga_history.json
	rm -f outputs/evolution/evolution_tree.json
	rm -f outputs/evolution/evolution_tree.tex
	rm -f outputs/evolution/evolution_tree.pdf
	rm -f outputs/evolution/evolution_tree.png
	rm -f outputs/evolution/reflector_message_*.txt
	rm -rf outputs/evolution/reflector_prompts
	rm -rf outputs/evolution/__pycache__
	rm -f outputs/evolution/_eval_deltas.json
	rm -f outputs/evolution/_eval_processor.py
	@echo "Evolution run outputs removed; config and directory preserved."
