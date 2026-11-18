// Todas as animações usam dados fictícios. Nenhum dado real de cliente.
(() => {
  'use strict';

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = (sel, root = document) => root.querySelector(sel);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  };
  const trim = (list, max) => { while (list.children.length > max) list.firstElementChild.remove(); };

  // Reveal ao rolar
  const revealIO = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add('in'); revealIO.unobserve(e.target); }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
  document.querySelectorAll('.reveal').forEach((n) => revealIO.observe(n));

  // Cada demo roda em laço enquanto visível; sai da tela, para.
  // `run(ctx)` checa ctx.alive para encerrar; `still()` desenha o estado final (movimento reduzido).
  function demo(root, run, still) {
    if (!root) return;
    const svgs = root.querySelectorAll('svg');
    if (reduced) { svgs.forEach((s) => s.pauseAnimations && s.pauseAnimations()); still(); return; }
    let ctx = null;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !ctx) {
        ctx = { alive: true };
        svgs.forEach((s) => s.unpauseAnimations && s.unpauseAnimations());
        const mine = ctx;
        (async () => {
          while (mine.alive) {
            try { await run(mine); }
            catch (err) { console.error('[demo]', root.dataset.demo || root.className, err); mine.alive = false; }
          }
        })();
      } else if (!e.isIntersecting && ctx) {
        ctx.alive = false; ctx = null;
        svgs.forEach((s) => s.pauseAnimations && s.pauseAnimations());
      }
    }, { threshold: 0.15 });
    io.observe(root);
  }
  const wait = async (ctx, ms) => { await sleep(ms); if (!ctx.alive) throw Object.assign(new Error('stop'), { stop: true }); };
  const guard = (fn) => async (ctx) => { try { await fn(ctx); } catch (e) { if (!e.stop) throw e; } };

  /* ---------- Hero: log do fluxo ---------- */
  const flowLog = $('#flowLog');
  const flowLines = [
    ['webhook', 'msg recebida · Lead C'],
    ['agente', 'intenção: aula experimental'],
    ['sql', 'SELECT turmas livres → 3 linhas'],
    ['resposta', 'enviada em 1,2 s'],
    ['agenda', 'varredura de promessas · 0 atrasadas'],
    ['agente', 'resumo diário gerado'],
    ['sql', 'INSERT log_execucao ok'],
  ];
  demo($('.hero-flow'), guard(async (ctx) => {
    for (const [k, v] of flowLines) {
      const t = new Date().toTimeString().slice(0, 8);
      flowLog.append(el('li', '', `<span class="mute">${t}</span> <b>${k}</b> ${v}`));
      trim(flowLog, 5);
      await wait(ctx, 1100);
    }
  }), () => {
    flowLines.slice(0, 5).forEach(([k, v]) => flowLog.append(el('li', '', `<b>${k}</b> ${v}`)));
  });

  /* ---------- Telas reais: sequência com zoom, cursor e legenda ---------- */
  // Coordenadas em % da imagem. focus = ponto de zoom; at = alvo do cursor; z = escala.
  const SHOTS = {
    agentes: [
      { src: 'agente-conversa', cap: 'Conversa real do sistema: o agente de IA atende o lead e agenda a aula experimental',
        focus: [62, 58], z: 1.35, at: [62, 61], call: '<b>Agente de IA</b> confirma o agendamento' },
      { src: 'agente-painel', cap: 'Painel do agente: leads atendidos, passagens de bastão e qualidade dos dados',
        focus: [55, 28], z: 1.3, at: [49, 25], call: '<b>Passagem de bastão</b> para a equipe comercial' },
    ],
    erp: [
      { src: 'erp-dashboard', cap: 'Dashboard do sistema de gestão: gestão, comercial e professores, com metas',
        focus: [50, 36], z: 1.32, at: [26, 34], call: '<b>Indicadores com meta</b> calculados no banco' },
      { src: 'erp-resumo', cap: 'Evolução de alunos, funil comercial e resumo por unidade',
        focus: [58, 78], z: 1.28, at: [26, 75], call: '<b>3 unidades</b> num painel só' },
    ],
    campanhas: [
      { src: 'campanhas-dashboard', cap: 'Campanhas de WhatsApp: envios, entregas, leituras, respostas e custo do mês', focus: [55, 45], z: 1.2, at: [22, 47], call: '<b>Custo do mês</b> e cota diária de envio' },
      { src: 'campanhas-lista', cap: 'Lista de campanhas com o funil de cada envio', focus: [58, 40], z: 1.25, at: [60, 44], call: '<b>Enviados, entregues, lidos e respostas</b> por campanha' },
      { src: 'campanhas-publico', cap: 'Nova campanha: público por unidade, via planilha, alunos ou leads', focus: [50, 40], z: 1.3, at: [33, 36], call: '<b>Público segmentado</b> por unidade' },
      { src: 'campanhas-envio-1', cap: 'Envio em andamento, acompanhado em tempo real', focus: [58, 35], z: 1.3, at: [27, 41], call: '<b>Funil de entrega</b> atualizando', quick: true },
      { src: 'campanhas-envio-2', cap: 'Envio em andamento, acompanhado em tempo real', focus: [58, 35], z: 1.3, quick: true, keep: true },
      { src: 'campanhas-envio-3', cap: 'Envio em andamento, acompanhado em tempo real', focus: [58, 35], z: 1.3, quick: true, keep: true },
      { src: 'campanhas-envio-4', cap: 'Campanha concluída, com conversão em leads e matrículas', focus: [58, 35], z: 1.3, keep: true, at: [40, 60], call: '<b>Concluída</b>: leads, experimentais e matrículas' },
      { src: 'campanhas-conversa', cap: 'Conversas da campanha: agente de IA responde quem recebeu o envio', focus: [60, 55], z: 1.25, at: [70, 55], call: '<b>Agente de IA</b> nas respostas' },
      { src: 'campanhas-conversao', cap: 'Conversão por campanha: leads, experimentais, matrículas e custo por matrícula', focus: [60, 40], z: 1.2, at: [73, 38], call: '<b>Taxa de conversão</b> por campanha' },
      { src: 'campanhas-analytics', cap: 'Analytics: custo estimado x real e taxas de entrega, leitura e resposta', focus: [60, 45], z: 1.2, at: [40, 62], call: '<b>Custo e taxas</b> por campanha' },
    ],
  };

  function shotDemo(root) {
    const list = SHOTS[root.dataset.shots];
    if (!list) { console.error('[shots] sequência desconhecida:', root.dataset.shots); return; }
    const stage = $('.shot-stage', root), cursor = $('.cursor', root), callout = $('.callout', root);
    const cap = $('.shot-cap', root), dotsBox = $('.shot-dots', root);
    const first = $('.frame', stage);
    const alt = first.querySelector('img').alt;
    const frames = list.map((f, i) => {
      if (i === 0) return first;
      const fig = el('figure', 'frame');
      const img = new Image(1440, 900);
      img.alt = alt; img.decoding = 'async'; img.loading = 'lazy';
      img.src = 'assets/telas/' + f.src + '.webp';
      img.onerror = () => console.error('[shots] imagem não carregou:', img.src);
      fig.append(img);
      stage.insertBefore(fig, cursor);
      return fig;
    });
    const dots = list.filter((f) => !f.keep).map(() => { const d = el('i'); dotsBox.append(d); return d; });
    let dotIdx = -1;
    const show = (i) => {
      frames.forEach((f, k) => f.classList.toggle('on', k === i));
      if (!list[i].keep) dotIdx++;
      dots.forEach((d, k) => d.classList.toggle('on', k === dotIdx));
      cap.textContent = list[i].cap;
    };
    const zoom = (i, on, instant) => {
      const f = list[i], img = frames[i].querySelector('img');
      if (instant) img.style.transition = 'none';
      img.style.transformOrigin = f.focus[0] + '% ' + f.focus[1] + '%';
      img.style.transform = on ? 'scale(' + f.z + ')' : 'scale(1)';
      if (instant) { void img.offsetWidth; img.style.transition = ''; }
    };
    // posição do alvo depois do zoom: o + s * (p - o)
    const mapped = (f, p, on) => [0, 1].map((k) => (on ? f.focus[k] + f.z * (p[k] - f.focus[k]) : p[k]));
    const place = (node, xy) => { node.style.left = xy[0] + '%'; node.style.top = xy[1] + '%'; };
    const placeCallout = (xy) => {
      const left = xy[0] > 60 ? null : xy[0] + 3, right = xy[0] > 60 ? 100 - xy[0] + 3 : null;
      callout.style.left = left === null ? 'auto' : left + '%';
      callout.style.right = right === null ? 'auto' : right + '%';
      callout.style.top = Math.min(Math.max(xy[1] + 4, 4), 82) + '%';
    };

    demo(root, guard(async (ctx) => {
      dotIdx = -1;
      for (let i = 0; i < list.length; i++) {
        const f = list[i];
        show(i);
        callout.classList.remove('on');
        // quadro em sequência (keep) já entra no mesmo zoom do anterior: sem "fantasma" no crossfade
        zoom(i, !!f.keep, true);
        const prev = i - 1;
        if (prev >= 0) setTimeout(() => zoom(prev, false, true), 900);
        await wait(ctx, f.quick ? 350 : 700);
        if (!f.keep) zoom(i, true);
        if (f.at) {
          const xy = mapped(f, f.at, true);
          cursor.classList.add('on');
          place(cursor, xy);
          await wait(ctx, 1200);
          cursor.classList.remove('click'); void cursor.offsetWidth; cursor.classList.add('click');
        }
        if (f.call) {
          callout.innerHTML = f.call;
          placeCallout(mapped(f, f.at || f.focus, true));
          callout.classList.add('on');
        }
        await wait(ctx, f.quick ? 900 : 3200);
      }
      cursor.classList.remove('on');
      callout.classList.remove('on');
      await wait(ctx, 400);
    }), () => {
      show(0);
      if (list[0].call) { callout.innerHTML = list[0].call; placeCallout(list[0].at || list[0].focus); callout.classList.add('on'); }
    });
  }
  document.querySelectorAll('.shot[data-shots]').forEach(shotDemo);

  /* ---------- 4. MCP ---------- */
  const term = $('#term');
  const termScript = [
    ['p', '$ agente → tools/list'],
    ['r', '  consultar_turmas · buscar_aluno · resumo_frequencia'],
    ['p', '$ agente → consultar_turmas {"instrumento":"bateria"}'],
    ['r', '  [{"turma":"T-07","vagas":3},{"turma":"T-12","vagas":1}]'],
    ['p', '$ agente → executar_sql "UPDATE alunos SET ..."'],
    ['x', '  ✗ recusado: conexão somente leitura'],
    ['p', '$ npm test'],
    ['ok', '  ✓ testes passando · ✓ evals de uso de ferramenta'],
  ];
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const termHTML = (lines, partial) =>
    lines.map(([c, t]) => `<span class="${c}">${esc(t)}</span>`).join('\n') + (partial !== undefined ? partial : '');
  demo($('.demo-mcp'), guard(async (ctx) => {
    const done = [];
    term.innerHTML = '<span class="caret"></span>';
    await wait(ctx, 600);
    for (const [c, t] of termScript) {
      if (c === 'p') {
        for (let i = 1; i <= t.length; i += 2) {
          term.innerHTML = termHTML(done) + (done.length ? '\n' : '') + `<span class="p">${esc(t.slice(0, i))}</span><span class="caret"></span>`;
          await wait(ctx, 28);
        }
        await wait(ctx, 250);
      } else {
        await wait(ctx, 450);
      }
      done.push([c, t]);
      term.innerHTML = termHTML(done, '\n<span class="caret"></span>');
    }
    await wait(ctx, 3500);
  }), () => { term.innerHTML = termHTML(termScript); });
})();
