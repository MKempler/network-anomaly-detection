"""The fault menu: what a visitor can break, and how each break is undone.

Only these faults exist, and only against containers carrying the
`lab.service` label; there is no free-form input. Every fault records what it
changed so `revert` restores exactly that.

- delay:  +400 ms on every packet the service sends (netem, from a sidecar
          sharing the service's network namespace)
- loss:   30% of the service's outgoing packets dropped (netem)
- cpu:    the service throttled to 1% of one core (5% left light services unaffected)
- pause:  the service frozen (docker pause), as if it hung
"""

from dataclasses import dataclass, field

NETEM_IMAGE = "nicolaka/netshoot:v0.13"
FAULTS = {
    "delay": "Slow network: +400 ms on every packet the service sends",
    "loss": "Packet loss: 30% of the service's outgoing packets dropped",
    "cpu": "CPU starvation: the service throttled to 1% of one core",
    "pause": "Hang: the service frozen mid-flight",
}
# redis-cart is left off: its only caller (cartservice) emits no traces, so nothing in
# the lab can observe it, and a fault nobody can see makes a poor demo.
TARGETS = ("frontend", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice",
           "productcatalogservice", "recommendationservice", "shippingservice", "adservice")
NETEM = {"delay": "delay 400ms", "loss": "loss 30%"}


@dataclass
class ActiveFault:
    service: str
    fault: str
    undo: dict = field(default_factory=dict)


class FaultInjector:
    def __init__(self, docker_client, label: str = "lab.service"):
        self.docker, self.label = docker_client, label

    def _container(self, service: str):
        if service not in TARGETS:
            raise ValueError(f"{service!r} is not a lab target")
        found = self.docker.containers.list(filters={"label": f"{self.label}={service}"})
        if not found:
            raise LookupError(f"no running container for {service}")
        return found[0]

    def _netem(self, container, args: str):
        self.docker.containers.run(NETEM_IMAGE, ["tc", "qdisc", *args.split()], remove=True,
                                   network_mode=f"container:{container.id}", cap_add=["NET_ADMIN"])

    def inject(self, service: str, fault: str) -> ActiveFault:
        if fault not in FAULTS:
            raise ValueError(f"{fault!r} is not on the fault menu")
        container = self._container(service)
        active = ActiveFault(service, fault)
        if fault in NETEM:
            self._netem(container, f"add dev eth0 root netem {NETEM[fault]}")
        elif fault == "cpu":
            host = container.attrs.get("HostConfig", {})
            active.undo = {"cpu_period": host.get("CpuPeriod") or 100000, "cpu_quota": host.get("CpuQuota") or -1}
            container.update(cpu_period=100000, cpu_quota=1000)
        elif fault == "pause":
            container.pause()
        return active

    def revert(self, active: ActiveFault):
        container = self._container(active.service)
        if active.fault in NETEM:
            self._netem(container, "del dev eth0 root")
        elif active.fault == "cpu":
            container.update(**active.undo)
        elif active.fault == "pause":
            container.unpause()
